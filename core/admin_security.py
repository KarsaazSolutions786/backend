"""
Admin Panel Security & RBAC
===========================

Comprehensive role-based access control system for admin panel including:
- JWT token verification with admin roles
- Permission decorators and dependencies
- Automatic audit logging
- Rate limiting and security measures
- 2FA support
"""

import jwt
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any, Callable
import os
import secrets
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
import logging
from functools import wraps
import asyncio
from contextlib import asynccontextmanager
from connect_db import get_db

from models.admin_models import AdminUser, AdminRole, AdminAuditLog, AuditAction

logger = logging.getLogger(__name__)

# Security configuration
JWT_SECRET_KEY = os.getenv("ADMIN_JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("ADMIN_JWT_EXPIRATION_HOURS", "8"))
LOCKOUT_ATTEMPTS = int(os.getenv("ADMIN_LOCKOUT_ATTEMPTS", "5"))
LOCKOUT_DURATION_MINUTES = int(os.getenv("ADMIN_LOCKOUT_DURATION", "30"))

# Rate limiting (in-memory store for simplicity, use Redis in production)
rate_limit_store: Dict[str, List[datetime]] = {}

security = HTTPBearer()

class AdminSecurityError(Exception):
    """Custom exception for admin security issues."""
    pass

class InsufficientPermissionsError(AdminSecurityError):
    """Raised when admin user lacks required permissions."""
    pass

class AccountLockedError(AdminSecurityError):
    """Raised when admin account is locked due to failed attempts."""
    pass

class TwoFactorRequiredError(AdminSecurityError):
    """Raised when 2FA is required but not provided."""
    pass

# JWT Token Operations
def create_admin_jwt_token(admin_user: AdminUser, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token for admin user with role information."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        "sub": str(admin_user.id),
        "email": admin_user.email,
        "name": admin_user.name,
        "role": admin_user.role.value,
        "iat": datetime.utcnow(),
        "exp": expire,
        "iss": "eindr-admin",
        "type": "admin_access"
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_admin_jwt_token(token: str) -> Dict[str, Any]:
    """Verify and decode admin JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Validate token type
        if payload.get("type") != "admin_access":
            raise jwt.InvalidTokenError("Invalid token type")
        
        # Check expiration
        if payload.get("exp", 0) < datetime.utcnow().timestamp():
            raise jwt.ExpiredSignatureError("Token has expired")
        
        return payload
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )

# Rate Limiting
def check_rate_limit(key: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
    """Check if request is within rate limits."""
    now = datetime.utcnow()
    window_start = now - timedelta(minutes=window_minutes)
    
    if key not in rate_limit_store:
        rate_limit_store[key] = []
    
    # Clean old requests
    rate_limit_store[key] = [
        req_time for req_time in rate_limit_store[key] 
        if req_time > window_start
    ]
    
    # Check limit
    if len(rate_limit_store[key]) >= max_requests:
        return False
    
    # Add current request
    rate_limit_store[key].append(now)
    return True

async def check_account_lockout(admin_user: AdminUser, db: AsyncSession) -> None:
    """Check if admin account is locked and handle lockout logic."""
    if admin_user.locked_until and admin_user.locked_until > datetime.utcnow():
        remaining_time = admin_user.locked_until - datetime.utcnow()
        raise AccountLockedError(
            f"Account locked for {remaining_time.seconds // 60} more minutes"
        )
    
    # Reset lock if expired
    if admin_user.locked_until and admin_user.locked_until <= datetime.utcnow():
        admin_user.locked_until = None
        admin_user.login_attempts = 0
        await db.commit()

async def handle_failed_login(admin_user: AdminUser, db: AsyncSession) -> None:
    """Handle failed login attempt with lockout logic."""
    admin_user.login_attempts += 1
    
    if admin_user.login_attempts >= LOCKOUT_ATTEMPTS:
        admin_user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        logger.warning(f"Admin account locked: {admin_user.email}")
    
    await db.commit()

async def handle_successful_login(admin_user: AdminUser, db: AsyncSession) -> None:
    """Handle successful login by resetting attempt counters."""
    admin_user.login_attempts = 0
    admin_user.locked_until = None
    admin_user.last_login_at = datetime.utcnow()
    await db.commit()

# 2FA Operations
def generate_2fa_secret() -> str:
    """Generate a new 2FA secret."""
    return pyotp.random_base32()

def generate_2fa_qr_code(admin_user: AdminUser, secret: str) -> str:
    """Generate QR code for 2FA setup."""
    issuer = "Eindr Admin"
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=admin_user.email,
        issuer_name=issuer
    )
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()

def verify_2fa_token(secret: str, token: str) -> bool:
    """Verify 2FA token."""
    totp = pyotp.TOTP(secret)
    return totp.verify(token, valid_window=1)  # Allow 30 second window

def generate_backup_codes(count: int = 10) -> List[str]:
    """Generate backup codes for 2FA recovery."""
    return [secrets.token_hex(4).upper() for _ in range(count)]

# Audit Logging
async def log_admin_action(
    admin_id: str,
    action: AuditAction,
    request: Request,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> None:
    """Log admin action for audit trail."""
    try:
        # Temporarily disabled for initial setup - would implement database logging
        logger.info(f"Admin action: {admin_id} - {action} - {description or 'No description'}")
        pass
    except Exception as e:
        logger.error(f"Failed to log admin action: {e}")

# Authentication Dependencies
async def get_current_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> AdminUser:
    """Get current authenticated admin user."""
    try:
        # Verify JWT token
        payload = verify_admin_jwt_token(credentials.credentials)
        admin_id = payload["sub"]
        
        # Rate limiting check
        if not check_rate_limit(f"admin:{admin_id}", max_requests=1000, window_minutes=60):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests"
            )
        
        # Get admin user from database
        db = next(get_db())
        try:
            result = db.execute(
                select(AdminUser).where(
                    AdminUser.id == admin_id,
                    AdminUser.is_active == True
                )
            )
            admin_user = result.scalar_one_or_none()
            
            if not admin_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Admin user not found or inactive"
                )
            
            return admin_user
        finally:
            db.close()
            
    except AccountLockedError as e:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Admin authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

def require_admin_roles(required_roles: List[AdminRole]):
    """Dependency factory to require specific admin roles."""
    async def check_role(
        current_admin: AdminUser = Depends(get_current_admin_user),
        request: Request = None
    ) -> AdminUser:
        if not current_admin.has_permission(required_roles):
            # Log unauthorized access attempt
            await log_admin_action(
                admin_id=str(current_admin.id),
                action=AuditAction.VIEW,
                request=request,
                success=False,
                error_message=f"Insufficient permissions. Required: {required_roles}, Has: {current_admin.role}"
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[role.value for role in required_roles]}"
            )
        
        return current_admin
    
    return check_role

# Convenience dependencies for common role checks
get_super_admin = require_admin_roles([AdminRole.SUPER_ADMIN])
get_support_or_super = require_admin_roles([AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT])
get_analyst_or_higher = require_admin_roles([AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT, AdminRole.ANALYST])
get_any_admin = require_admin_roles([AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT, AdminRole.ANALYST, AdminRole.READ_ONLY])

# Audit Logging Decorator
def audit_action(
    action: AuditAction,
    target_type: Optional[str] = None,
    description: Optional[str] = None
):
    """Decorator to automatically audit admin actions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and current admin from function arguments
            request = None
            current_admin = None
            target_id = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                elif isinstance(arg, AdminUser):
                    current_admin = arg
            
            # Check kwargs for these objects
            if not request:
                request = kwargs.get('request')
            if not current_admin:
                current_admin = kwargs.get('current_admin')
            
            # Extract target_id from path parameters or request body
            if hasattr(func, '__annotations__'):
                for param_name, param_type in func.__annotations__.items():
                    if param_name in kwargs and (
                        param_name.endswith('_id') or 
                        param_name in ['user_id', 'notification_id', 'flag_id']
                    ):
                        target_id = str(kwargs[param_name])
                        break
            
            success = True
            error_message = None
            result = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # Log the action if we have the required info
                if current_admin and request:
                    await log_admin_action(
                        admin_id=str(current_admin.id),
                        action=action,
                        request=request,
                        target_type=target_type,
                        target_id=target_id,
                        description=description or f"{action.value.title()} {target_type}" if target_type else None,
                        success=success,
                        error_message=error_message
                    )
        
        return wrapper
    return decorator

# Security Headers Middleware
class AdminSecurityMiddleware:
    """Middleware to add security headers for admin endpoints."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/admin"):
            async def send_with_security_headers(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Add security headers
                    headers.update({
                        b"x-content-type-options": b"nosniff",
                        b"x-frame-options": b"DENY",
                        b"x-xss-protection": b"1; mode=block",
                        b"strict-transport-security": b"max-age=31536000; includeSubDomains",
                        b"cache-control": b"no-cache, no-store, must-revalidate",
                        b"pragma": b"no-cache",
                        b"expires": b"0"
                    })
                    
                    message["headers"] = list(headers.items())
                
                await send(message)
            
            await self.app(scope, receive, send_with_security_headers)
        else:
            await self.app(scope, receive, send)

# Password Utilities
def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength for admin accounts."""
    errors = []
    score = 0
    
    if len(password) < 12:
        errors.append("Password must be at least 12 characters long")
    else:
        score += 1
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    else:
        score += 1
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    else:
        score += 1
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one number")
    else:
        score += 1
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("Password must contain at least one special character")
    else:
        score += 1
    
    strength = "Very Weak"
    if score >= 4:
        strength = "Strong"
    elif score >= 3:
        strength = "Medium"
    elif score >= 2:
        strength = "Weak"
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "score": score,
        "strength": strength
    }

# Session Management
class AdminSessionManager:
    """Manage admin user sessions and concurrent login limits."""
    
    active_sessions: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def add_session(cls, admin_id: str, token: str, request: Request):
        """Add new admin session."""
        if admin_id not in cls.active_sessions:
            cls.active_sessions[admin_id] = {}
        
        cls.active_sessions[admin_id][token] = {
            "created_at": datetime.utcnow(),
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "last_activity": datetime.utcnow()
        }
    
    @classmethod
    def remove_session(cls, admin_id: str, token: str):
        """Remove admin session."""
        if admin_id in cls.active_sessions and token in cls.active_sessions[admin_id]:
            del cls.active_sessions[admin_id][token]
            if not cls.active_sessions[admin_id]:
                del cls.active_sessions[admin_id]
    
    @classmethod
    def update_activity(cls, admin_id: str, token: str):
        """Update last activity for session."""
        if admin_id in cls.active_sessions and token in cls.active_sessions[admin_id]:
            cls.active_sessions[admin_id][token]["last_activity"] = datetime.utcnow()
    
    @classmethod
    def get_active_sessions(cls, admin_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for admin user."""
        return list(cls.active_sessions.get(admin_id, {}).values())
    
    @classmethod
    def cleanup_expired_sessions(cls, max_age_hours: int = 24):
        """Clean up expired sessions."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for admin_id in list(cls.active_sessions.keys()):
            sessions = cls.active_sessions[admin_id]
            for token in list(sessions.keys()):
                if sessions[token]["last_activity"] < cutoff:
                    del sessions[token]
            
            if not sessions:
                del cls.active_sessions[admin_id]

# Add async database session manager
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Create async engine and session
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:SFXMbavWxAsHrPlVMZXjsNvRUqZNajRz@metro.proxy.rlwy.net:54916/railway")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    echo=False
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@asynccontextmanager
async def get_database():
    """Async database session dependency."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close() 