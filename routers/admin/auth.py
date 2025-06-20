"""
Admin Authentication Router
===========================

Comprehensive admin authentication system including:
- Admin login with JWT tokens
- Two-factor authentication (2FA) 
- Password management and security
- Session management
- Account lockout protection
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.admin_security import (
    create_admin_jwt_token, verify_admin_jwt_token, get_current_admin_user,
    check_account_lockout, handle_failed_login, handle_successful_login,
    generate_2fa_secret, generate_2fa_qr_code, verify_2fa_token, generate_backup_codes,
    validate_password_strength, AdminSessionManager, log_admin_action,
    security, AccountLockedError, TwoFactorRequiredError
)
from models.admin_models import AdminUser, AdminRole, AuditAction
from connect_db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/auth",
    tags=["Admin - Authentication"],
    responses={
        401: {"description": "Unauthorized - Invalid credentials"},
        423: {"description": "Account locked due to failed attempts"},
        500: {"description": "Internal server error"}
    }
)

# Pydantic models for request/response
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)
    two_fa_token: Optional[str] = Field(None, description="2FA token if enabled")
    remember_me: bool = Field(default=False, description="Extend token expiration")

class LoginResponse(BaseModel):
    success: bool
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    admin_user: Dict[str, Any]
    requires_2fa: bool = False
    session_info: Dict[str, Any]

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=12)
    confirm_password: str

class Setup2FAResponse(BaseModel):
    success: bool
    secret: str
    qr_code: str
    backup_codes: List[str]

class Verify2FARequest(BaseModel):
    token: str = Field(..., min_length=6, max_length=6)

class RefreshTokenRequest(BaseModel):
    refresh_token: Optional[str] = None

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Admin Login",
    description="Authenticate admin user with email/password and optional 2FA"
)
async def admin_login(
    login_data: LoginRequest,
    request: Request,
    response: Response
) -> LoginResponse:
    """
    Authenticate admin user and return JWT token.
    
    Supports:
    - Email/password authentication
    - Two-factor authentication (2FA)
    - Account lockout protection
    - Session management
    """
    try:
        # Use real database authentication
        db = next(get_db())
        try:
            # Find admin user by email
            result = db.execute(
                select(AdminUser).where(
                    AdminUser.email == login_data.email.lower(),
                    AdminUser.is_active == True
                )
            )
            admin_user = result.scalar_one_or_none()
            
            if not admin_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Verify password
            if not admin_user.verify_password(login_data.password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Generate JWT token
            expires_delta = timedelta(hours=24) if login_data.remember_me else timedelta(hours=8)
            access_token = create_admin_jwt_token(admin_user, expires_delta)
            
            # Add session to manager
            AdminSessionManager.add_session(str(admin_user.id), access_token, request)
            
            # Update last login
            admin_user.last_login_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Admin login successful: {admin_user.email}")
            
            # Create response
            admin_dict = {
                'id': str(admin_user.id),
                'name': admin_user.name,
                'email': admin_user.email,
                'role': admin_user.role.value,
                'two_fa_enabled': admin_user.two_fa_enabled,
                'is_active': admin_user.is_active,
                'last_login_at': admin_user.last_login_at.isoformat() if admin_user.last_login_at else None,
                'created_at': admin_user.created_at.isoformat() if admin_user.created_at else None,
                'updated_at': admin_user.updated_at.isoformat() if admin_user.updated_at else None
            }
            
            return LoginResponse(
                success=True,
                access_token=access_token,
                expires_in=int(expires_delta.total_seconds()),
                admin_user=admin_dict,
                requires_2fa=False,
                session_info={
                    "login_time": datetime.utcnow().isoformat(),
                    "ip_address": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                    "remember_me": login_data.remember_me
                }
            )
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@router.post(
    "/logout",
    response_model=Dict[str, Any],
    summary="Admin Logout",
    description="Logout admin user and invalidate session"
)
async def admin_logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Logout admin user and invalidate the current session.
    """
    try:
        # Remove session
        AdminSessionManager.remove_session(str(current_admin.id), credentials.credentials)
        
        # Log logout
        await log_admin_action(
            admin_id=str(current_admin.id),
            action=AuditAction.LOGOUT,
            request=request,
            success=True,
            description="Admin logout"
        )
        
        logger.info(f"Admin logout: {current_admin.email}")
        
        return {
            "success": True,
            "message": "Logged out successfully",
            "logged_out_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Admin logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service error"
        )

@router.post(
    "/change-password",
    response_model=Dict[str, Any],
    summary="Change Password",
    description="Change admin user password with validation"
)
async def change_password(
    password_data: ChangePasswordRequest,
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Change admin user password with comprehensive validation.
    """
    try:
        # Validate current password
        if not current_admin.verify_password(password_data.current_password):
            await log_admin_action(
                admin_id=str(current_admin.id),
                action=AuditAction.UPDATE,
                request=request,
                target_type="password",
                success=False,
                error_message="Invalid current password"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid current password"
            )
        
        # Validate new password confirmation
        if password_data.new_password != password_data.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password and confirmation do not match"
            )
        
        # Validate password strength
        strength_check = validate_password_strength(password_data.new_password)
        if not strength_check["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Password does not meet security requirements",
                    "errors": strength_check["errors"],
                    "strength": strength_check["strength"]
                }
            )
        
        # Update password
        async with get_db() as db:
            current_admin.set_password(password_data.new_password)
            await db.commit()
        
        # Log password change
        await log_admin_action(
            admin_id=str(current_admin.id),
            action=AuditAction.UPDATE,
            request=request,
            target_type="password",
            success=True,
            description="Password changed successfully"
        )
        
        logger.info(f"Password changed for admin: {current_admin.email}")
        
        return {
            "success": True,
            "message": "Password changed successfully",
            "password_strength": strength_check["strength"],
            "changed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change service error"
        )

@router.post(
    "/setup-2fa",
    response_model=Setup2FAResponse,
    summary="Setup 2FA",
    description="Setup two-factor authentication for admin account"
)
async def setup_2fa(
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Setup2FAResponse:
    """
    Setup two-factor authentication for admin account.
    
    Returns QR code and backup codes for 2FA setup.
    """
    try:
        if current_admin.two_fa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is already enabled for this account"
            )
        
        # Generate 2FA secret and QR code
        secret = generate_2fa_secret()
        qr_code = generate_2fa_qr_code(current_admin, secret)
        backup_codes = generate_backup_codes()
        
        # Store secret and backup codes in database
        async with get_db() as db:
            current_admin.two_fa_secret = secret
            current_admin.backup_codes = backup_codes
            # Don't enable 2FA yet - wait for verification
            await db.commit()
        
        # Log 2FA setup
        await log_admin_action(
            admin_id=str(current_admin.id),
            action=AuditAction.CREATE,
            request=request,
            target_type="2fa_setup",
            success=True,
            description="2FA setup initiated"
        )
        
        logger.info(f"2FA setup initiated for admin: {current_admin.email}")
        
        return Setup2FAResponse(
            success=True,
            secret=secret,
            qr_code=qr_code,
            backup_codes=backup_codes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA setup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="2FA setup service error"
        )

@router.post(
    "/verify-2fa",
    response_model=Dict[str, Any],
    summary="Verify 2FA Setup",
    description="Verify 2FA token to complete setup"
)
async def verify_2fa_setup(
    verify_data: Verify2FARequest,
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Verify 2FA token to complete 2FA setup.
    """
    try:
        if current_admin.two_fa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is already enabled for this account"
            )
        
        if not current_admin.two_fa_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA setup not initiated. Please setup 2FA first."
            )
        
        # Verify token
        if not verify_2fa_token(current_admin.two_fa_secret, verify_data.token):
            await log_admin_action(
                admin_id=str(current_admin.id),
                action=AuditAction.UPDATE,
                request=request,
                target_type="2fa_verification",
                success=False,
                error_message="Invalid 2FA token during setup"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA token"
            )
        
        # Enable 2FA
        async with get_db() as db:
            current_admin.two_fa_enabled = True
            await db.commit()
        
        # Log 2FA activation
        await log_admin_action(
            admin_id=str(current_admin.id),
            action=AuditAction.UPDATE,
            request=request,
            target_type="2fa_activation",
            success=True,
            description="2FA enabled successfully"
        )
        
        logger.info(f"2FA enabled for admin: {current_admin.email}")
        
        return {
            "success": True,
            "message": "2FA enabled successfully",
            "enabled_at": datetime.utcnow().isoformat(),
            "backup_codes_count": len(current_admin.backup_codes) if current_admin.backup_codes else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="2FA verification service error"
        )

@router.post(
    "/disable-2fa",
    response_model=Dict[str, Any],
    summary="Disable 2FA",
    description="Disable two-factor authentication for admin account"
)
async def disable_2fa(
    password_data: Dict[str, str],
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Disable two-factor authentication for admin account.
    Requires password confirmation for security.
    """
    try:
        if not current_admin.two_fa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is not enabled for this account"
            )
        
        # Verify password
        password = password_data.get("password")
        if not password or not current_admin.verify_password(password):
            await log_admin_action(
                admin_id=str(current_admin.id),
                action=AuditAction.UPDATE,
                request=request,
                target_type="2fa_disable",
                success=False,
                error_message="Invalid password for 2FA disable"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password"
            )
        
        # Disable 2FA
        async with get_db() as db:
            current_admin.two_fa_enabled = False
            current_admin.two_fa_secret = None
            current_admin.backup_codes = None
            await db.commit()
        
        # Log 2FA disabling
        await log_admin_action(
            admin_id=str(current_admin.id),
            action=AuditAction.UPDATE,
            request=request,
            target_type="2fa_disable",
            success=True,
            description="2FA disabled"
        )
        
        logger.info(f"2FA disabled for admin: {current_admin.email}")
        
        return {
            "success": True,
            "message": "2FA disabled successfully",
            "disabled_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA disable error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="2FA disable service error"
        )

@router.get(
    "/profile",
    response_model=Dict[str, Any],
    summary="Get Admin Profile",
    description="Get current admin user profile information"
)
async def get_admin_profile(
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Get current admin user profile information.
    """
    try:
        # Get active sessions
        active_sessions = AdminSessionManager.get_active_sessions(str(current_admin.id))
        
        # Create profile dict manually to avoid issues with mock admin user
        profile_dict = {
            'id': str(current_admin.id),
            'name': current_admin.name,
            'email': current_admin.email,
            'role': current_admin.role.value,
            'two_fa_enabled': current_admin.two_fa_enabled,
            'is_active': current_admin.is_active,
            'last_login_at': current_admin.last_login_at.isoformat() if current_admin.last_login_at else None,
            'created_at': current_admin.created_at.isoformat() if hasattr(current_admin, 'created_at') and current_admin.created_at else None,
            'updated_at': current_admin.updated_at.isoformat() if hasattr(current_admin, 'updated_at') and current_admin.updated_at else None
        }
        
        return {
            "success": True,
            "profile": profile_dict,
            "security_info": {
                "two_fa_enabled": current_admin.two_fa_enabled,
                "last_login": current_admin.last_login_at.isoformat() if current_admin.last_login_at else None,
                "password_changed_at": current_admin.password_changed_at.isoformat() if hasattr(current_admin, 'password_changed_at') and current_admin.password_changed_at else None,
                "active_sessions_count": len(active_sessions),
                "account_locked": hasattr(current_admin, 'locked_until') and current_admin.locked_until is not None and current_admin.locked_until > datetime.utcnow()
            },
            "permissions": {
                "role": current_admin.role.value,
                "can_manage_users": current_admin.role in [AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT],
                "can_view_analytics": current_admin.role in [AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT, AdminRole.ANALYST],
                "can_manage_features": current_admin.role == AdminRole.SUPER_ADMIN,
                "can_send_notifications": current_admin.role in [AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT]
            }
        }
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile retrieval service error"
        )

@router.get(
    "/sessions",
    response_model=Dict[str, Any],
    summary="Get Active Sessions",
    description="Get list of active admin sessions"
)
async def get_active_sessions(
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Get list of active admin sessions for the current user.
    """
    try:
        active_sessions = AdminSessionManager.get_active_sessions(str(current_admin.id))
        
        return {
            "success": True,
            "active_sessions": active_sessions,
            "total_sessions": len(active_sessions),
            "current_session": {
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        }
        
    except Exception as e:
        logger.error(f"Sessions retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sessions retrieval service error"
        )

@router.post(
    "/refresh-token",
    response_model=Dict[str, Any],
    summary="Refresh Access Token",
    description="Refresh admin access token before expiration"
)
async def refresh_access_token(
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin_user)
) -> Dict[str, Any]:
    """
    Refresh admin access token to extend session.
    """
    try:
        # Generate new token
        new_token = create_admin_jwt_token(current_admin)
        
        # Update session manager
        AdminSessionManager.add_session(str(current_admin.id), new_token, request)
        
        # Log token refresh
        await log_admin_action(
            admin_id=str(current_admin.id),
            action=AuditAction.UPDATE,
            request=request,
            target_type="token_refresh",
            success=True,
            description="Access token refreshed"
        )
        
        return {
            "success": True,
            "access_token": new_token,
            "token_type": "bearer",
            "expires_in": 8 * 3600,  # 8 hours in seconds
            "refreshed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh service error"
        ) 