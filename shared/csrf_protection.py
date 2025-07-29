"""
CSRF Protection for Eindr Microservices

This module provides Cross-Site Request Forgery (CSRF) protection for all
state-changing operations across the microservices platform.
"""

import secrets
import hashlib
import hmac
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis
import os
import json
from .redis_manager import get_redis_manager

logger = logging.getLogger(__name__)

class CSRFConfig:
    """Configuration for CSRF protection"""
    
    # Token settings
    TOKEN_LENGTH = int(os.getenv("CSRF_TOKEN_LENGTH", "32"))
    TOKEN_EXPIRY_HOURS = int(os.getenv("CSRF_TOKEN_EXPIRY_HOURS", "24"))
    SECRET_KEY = os.getenv("CSRF_SECRET_KEY", os.getenv("SECRET_KEY", "your-secret-key"))
    
    # Headers and cookies
    TOKEN_HEADER_NAME = "X-CSRF-Token"
    TOKEN_COOKIE_NAME = "csrf_token"
    
    # Methods that require CSRF protection
    PROTECTED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    
    # Paths to exclude from CSRF protection
    EXCLUDED_PATHS = {
        "/docs",
        "/openapi.json",
        "/health",
        "/metrics",
        "/auth/login",  # Initial login doesn't have CSRF token yet
        "/auth/register"  # Initial registration doesn't have CSRF token yet
    }
    
    # Origins that are allowed without CSRF (be very careful with this)
    TRUSTED_ORIGINS: Set[str] = set()
    
    # Enable double submit cookie pattern
    DOUBLE_SUBMIT_COOKIE = os.getenv("CSRF_DOUBLE_SUBMIT_COOKIE", "true").lower() == "true"
    
    # Use Redis for token storage
    USE_REDIS = os.getenv("CSRF_USE_REDIS", "true").lower() == "true"

class CSRFTokenManager:
    """Manages CSRF token generation, validation, and storage"""
    
    def __init__(self, config: CSRFConfig = None, redis_client: Optional[redis.Redis] = None):
        self.config = config or CSRFConfig()
        self.redis_manager = get_redis_manager() if redis_client is None else None
        self.redis_client = redis_client or (self.redis_manager.client if self.redis_manager and self.redis_manager.is_available else None)
        self.redis_prefix = "csrf_token:"
        
    def generate_token(self, session_id: str = None) -> str:
        """
        Generate a cryptographically secure CSRF token
        
        Args:
            session_id: Optional session identifier for token binding
            
        Returns:
            CSRF token string
        """
        # Generate random token
        random_token = secrets.token_urlsafe(self.config.TOKEN_LENGTH)
        
        # If session binding is enabled, create HMAC
        if session_id and self.config.SECRET_KEY:
            message = f"{random_token}:{session_id}:{datetime.utcnow().isoformat()}"
            signature = hmac.new(
                self.config.SECRET_KEY.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            return f"{random_token}.{signature}"
        
        return random_token
    
    def validate_token(self, token: str, session_id: str = None, customer_id: int = None) -> bool:
        """
        Validate a CSRF token
        
        Args:
            token: CSRF token to validate
            session_id: Session identifier for bound tokens
            customer_id: Customer ID for user-specific validation
            
        Returns:
            True if token is valid
        """
        if not token:
            return False
        
        # Check Redis storage if enabled
        if self.redis_client and self.config.USE_REDIS:
            return self._validate_stored_token(token, customer_id)
        
        # Validate HMAC-signed token
        if "." in token and session_id:
            return self._validate_signed_token(token, session_id)
        
        # For stateless tokens, just check format and length
        return len(token) >= self.config.TOKEN_LENGTH
    
    def store_token(self, token: str, customer_id: int, expires_in_hours: int = None) -> bool:
        """
        Store CSRF token for validation
        
        Args:
            token: CSRF token to store
            customer_id: Customer ID
            expires_in_hours: Token expiry time
            
        Returns:
            True if stored successfully
        """
        if not self.redis_client:
            return True  # No storage needed for stateless tokens
        
        expiry_hours = expires_in_hours or self.config.TOKEN_EXPIRY_HOURS
        ttl = expiry_hours * 3600  # Convert to seconds
        
        token_data = {
            "customer_id": customer_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=expiry_hours)).isoformat()
        }
        
        key = f"{self.redis_prefix}{token}"
        try:
            if self.redis_manager:
                success = self.redis_manager.set(key, json.dumps(token_data), ex=ttl)
                return success
            else:
                self.redis_client.setex(key, ttl, json.dumps(token_data))
                return True
        except Exception as e:
            logger.error(f"Failed to store CSRF token: {e}")
            return False
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a CSRF token
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked successfully
        """
        if not self.redis_client:
            return True
        
        key = f"{self.redis_prefix}{token}"
        try:
            if self.redis_manager:
                success = self.redis_manager.delete(key)
                return success
            else:
                self.redis_client.delete(key)
                return True
        except Exception as e:
            logger.error(f"Failed to revoke CSRF token: {e}")
            return False
    
    def revoke_user_tokens(self, customer_id: int) -> int:
        """
        Revoke all CSRF tokens for a user
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Number of tokens revoked
        """
        if not self.redis_client:
            return 0
        
        try:
            # Find all tokens for this user
            pattern = f"{self.redis_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            revoked_count = 0
            for key in keys:
                try:
                    token_data = self.redis_client.get(key)
                    if token_data:
                        data = json.loads(token_data)
                        if data.get("customer_id") == customer_id:
                            self.redis_client.delete(key)
                            revoked_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return revoked_count
        except Exception as e:
            logger.error(f"Failed to revoke user CSRF tokens: {e}")
            return 0
    
    def _validate_stored_token(self, token: str, customer_id: int = None) -> bool:
        """Validate token stored in Redis"""
        key = f"{self.redis_prefix}{token}"
        
        try:
            token_data = self.redis_client.get(key)
            if not token_data:
                return False
            
            data = json.loads(token_data)
            
            # Check expiry
            expires_at = datetime.fromisoformat(data["expires_at"])
            if datetime.utcnow() > expires_at:
                self.redis_client.delete(key)
                return False
            
            # Check customer ID if provided
            if customer_id and data.get("customer_id") != customer_id:
                return False
            
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid CSRF token data: {e}")
            self.redis_client.delete(key)
            return False
    
    def _validate_signed_token(self, token: str, session_id: str) -> bool:
        """Validate HMAC-signed token"""
        try:
            token_part, signature = token.rsplit(".", 1)
            
            # Reconstruct message (we don't know the exact timestamp, so we check recent ones)
            current_time = datetime.utcnow()
            
            # Check tokens created within the last 24 hours
            for hours_ago in range(self.config.TOKEN_EXPIRY_HOURS):
                check_time = current_time - timedelta(hours=hours_ago)
                message = f"{token_part}:{session_id}:{check_time.isoformat()}"
                
                expected_signature = hmac.new(
                    self.config.SECRET_KEY.encode(),
                    message.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                if hmac.compare_digest(signature, expected_signature):
                    return True
            
            return False
        except (ValueError, AttributeError):
            return False

class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware"""
    
    def __init__(self, app, config: CSRFConfig = None, token_manager: CSRFTokenManager = None):
        super().__init__(app)
        self.config = config or CSRFConfig()
        self.token_manager = token_manager or CSRFTokenManager(self.config)
    
    async def dispatch(self, request: Request, call_next):
        # Skip CSRF protection for excluded paths
        if self._should_skip_csrf(request):
            return await call_next(request)
        
        # Skip for safe methods
        if request.method not in self.config.PROTECTED_METHODS:
            return await call_next(request)
        
        # Check for CSRF token
        csrf_token = self._extract_csrf_token(request)
        if not csrf_token:
            return self._csrf_error("CSRF token missing")
        
        # Validate token
        if not self._validate_csrf_token(request, csrf_token):
            return self._csrf_error("Invalid CSRF token")
        
        # Add CSRF token to request for use in handlers
        request.state.csrf_token = csrf_token
        
        response = await call_next(request)
        return response
    
    def _should_skip_csrf(self, request: Request) -> bool:
        """Check if CSRF protection should be skipped"""
        path = request.url.path
        
        # Check excluded paths
        for excluded_path in self.config.EXCLUDED_PATHS:
            if path.startswith(excluded_path):
                return True
        
        # Check trusted origins
        origin = request.headers.get("Origin")
        if origin in self.config.TRUSTED_ORIGINS:
            return True
        
        # Skip for API documentation and health checks
        if any(skip in path for skip in ["/docs", "/openapi", "/health", "/metrics"]):
            return True
        
        return False
    
    def _extract_csrf_token(self, request: Request) -> Optional[str]:
        """Extract CSRF token from request"""
        # Try header first
        token = request.headers.get(self.config.TOKEN_HEADER_NAME)
        if token:
            return token
        
        # Try cookie if double submit pattern is enabled
        if self.config.DOUBLE_SUBMIT_COOKIE:
            token = request.cookies.get(self.config.TOKEN_COOKIE_NAME)
            if token:
                return token
        
        # Try form data for traditional forms
        if request.method == "POST" and "application/x-www-form-urlencoded" in request.headers.get("content-type", ""):
            # This would require parsing the form, which is more complex
            # For now, we'll rely on header/cookie methods
            pass
        
        return None
    
    def _validate_csrf_token(self, request: Request, token: str) -> bool:
        """Validate CSRF token"""
        # Get session/customer info if available
        session_id = request.cookies.get("session_id")
        customer_id = getattr(request.state, "customer_id", None)
        
        return self.token_manager.validate_token(token, session_id, customer_id)
    
    def _csrf_error(self, message: str) -> JSONResponse:
        """Return CSRF error response"""
        logger.warning(f"CSRF protection triggered: {message}")
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": message, "error_code": "CSRF_PROTECTION"}
        )

class CSRFProtection:
    """Main CSRF protection service"""
    
    def __init__(self, config: CSRFConfig = None, redis_client: Optional[redis.Redis] = None):
        self.config = config or CSRFConfig()
        self.token_manager = CSRFTokenManager(self.config, redis_client)
    
    def generate_token(self, customer_id: int, session_id: str = None) -> Dict[str, str]:
        """
        Generate CSRF token for a user
        
        Args:
            customer_id: Customer ID
            session_id: Optional session ID
            
        Returns:
            Dictionary with token and cookie info
        """
        token = self.token_manager.generate_token(session_id)
        
        # Store token if Redis is available
        if self.config.USE_REDIS:
            self.token_manager.store_token(token, customer_id)
        
        return {
            "token": token,
            "header_name": self.config.TOKEN_HEADER_NAME,
            "cookie_name": self.config.TOKEN_COOKIE_NAME,
            "expires_in_hours": self.config.TOKEN_EXPIRY_HOURS
        }
    
    def validate_request(self, request: Request, customer_id: int = None) -> bool:
        """
        Validate CSRF protection for a request
        
        Args:
            request: FastAPI request object
            customer_id: Customer ID for validation
            
        Returns:
            True if request is valid
        """
        # Skip for safe methods
        if request.method not in self.config.PROTECTED_METHODS:
            return True
        
        # Extract token
        token = request.headers.get(self.config.TOKEN_HEADER_NAME)
        if not token and self.config.DOUBLE_SUBMIT_COOKIE:
            token = request.cookies.get(self.config.TOKEN_COOKIE_NAME)
        
        if not token:
            return False
        
        # Validate token
        session_id = request.cookies.get("session_id")
        return self.token_manager.validate_token(token, session_id, customer_id)
    
    def require_csrf_token(self, customer_id: int = None):
        """
        Dependency function to require CSRF token
        
        Args:
            customer_id: Customer ID for validation
        """
        def csrf_dependency(request: Request):
            if not self.validate_request(request, customer_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="CSRF token required"
                )
            return True
        
        return csrf_dependency
    
    def logout_revoke_tokens(self, customer_id: int) -> int:
        """
        Revoke all CSRF tokens when user logs out
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Number of tokens revoked
        """
        return self.token_manager.revoke_user_tokens(customer_id)

def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client for CSRF token storage with enhanced error handling"""
    redis_manager = get_redis_manager()
    if redis_manager.is_available:
        return redis_manager.client
    return None

# Global instances
_csrf_redis_client = get_redis_client()
csrf_config = CSRFConfig()
csrf_protection = CSRFProtection(csrf_config, _csrf_redis_client)

# Convenience functions
def generate_csrf_token(customer_id: int, session_id: str = None) -> Dict[str, str]:
    """Generate CSRF token for user"""
    return csrf_protection.generate_token(customer_id, session_id)

def validate_csrf_request(request: Request, customer_id: int = None) -> bool:
    """Validate CSRF protection for request"""
    return csrf_protection.validate_request(request, customer_id)

def require_csrf_token(customer_id: int = None):
    """Dependency to require CSRF token"""
    return csrf_protection.require_csrf_token(customer_id)

# Decorator for route protection
def csrf_protect(customer_id_field: str = "customer_id"):
    """
    Decorator to protect routes with CSRF
    
    Args:
        customer_id_field: Field name in route arguments containing customer_id
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not available"
                )
            
            customer_id = kwargs.get(customer_id_field)
            if not validate_csrf_request(request, customer_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="CSRF protection failed"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator