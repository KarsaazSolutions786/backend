"""
Shared Authentication Utilities for Eindr Microservices

This module provides secure, consistent JWT validation and authentication
utilities that all microservices can use.
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import functools

logger = logging.getLogger(__name__)

security = HTTPBearer()

class AuthConfig:
    """Centralized authentication configuration"""
    SECRET_KEY = os.getenv("JWT_SECRET", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
    
    # Validate that we're not using a weak default key in production
    @functools.lru_cache()
    def validate_production_config(self):
        if os.getenv("ENVIRONMENT") == "production" and self.SECRET_KEY in [
            "your-secret-key-here", 
            "your-secret-key-here-change-in-production",
            "eindr-super-secret-key-change-in-production-123456789"
        ]:
            raise ValueError("Production environment requires a secure JWT_SECRET")

class SecureJWTValidator:
    """Secure JWT token validator with proper signature verification"""
    
    def __init__(self):
        self.secret_key = AuthConfig.SECRET_KEY
        self.algorithm = AuthConfig.ALGORITHM
    
    def verify_token(self, token: str, expected_type: str = "access") -> Dict:
        """
        Securely verify JWT token with signature validation
        
        Args:
            token: JWT token string
            expected_type: Expected token type (access or refresh)
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid, expired, or malformed
        """
        try:
            # Verify token signature and decode payload
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_signature": True, "verify_exp": True}
            )
            
            # Validate token type (optional for backward compatibility)
            token_type = payload.get("type")
            if token_type and token_type != expected_type:
                logger.warning(f"Invalid token type: expected {expected_type}, got {token_type}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Validate required fields
            if not payload.get("sub"):
                logger.warning("Token missing subject (sub) field")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing subject",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Check if token is expired (redundant with verify_exp but explicit)
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                logger.warning("Token has expired")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Unexpected error validating token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_customer_id_from_token(self, token: str) -> int:
        """
        Extract and validate customer ID from JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Customer ID as integer
        """
        payload = self.verify_token(token)
        customer_id = payload.get("sub")
        
        try:
            return int(customer_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid customer ID in token: {customer_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid customer ID in token",
                headers={"WWW-Authenticate": "Bearer"},
            )

# Global JWT validator instance
jwt_validator = SecureJWTValidator()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    FastAPI dependency to get current authenticated customer ID
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        Customer ID as integer
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    return jwt_validator.get_customer_id_from_token(token)

def get_current_user_payload(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    FastAPI dependency to get current authenticated customer's full token payload
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        Full token payload as dictionary
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    return jwt_validator.verify_token(token)

def require_permissions(required_permissions: List[str]):
    """
    Decorator for endpoints that require specific permissions
    
    Args:
        required_permissions: List of required permission strings
    """
    def permission_checker(payload: Dict = Depends(get_current_user_payload)):
        user_permissions = payload.get("permissions", [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions: {permission} required"
                )
        
        return payload
    
    return permission_checker

def verify_user_owns_resource(resource_customer_id: int, current_customer: int = Depends(get_current_user)):
    """
    Verify that the current customer owns the requested resource
    
    Args:
        resource_customer_id: Customer ID associated with the resource
        current_customer: Current authenticated customer ID
        
    Raises:
        HTTPException: If user doesn't own the resource
    """
    if current_customer != resource_customer_id:
        logger.warning(f"User {current_customer} attempted to access resource owned by {resource_customer_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only access your own resources"
        )

class TokenBlacklist:
    """
    Simple in-memory token blacklist for logout/revocation
    In production, this should use Redis or a database
    """
    def __init__(self):
        self._blacklisted_tokens = set()
    
    def add_token(self, token: str):
        """Add token to blacklist"""
        self._blacklisted_tokens.add(token)
    
    def is_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        return token in self._blacklisted_tokens
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens from blacklist (should be run periodically)"""
        # In a real implementation, you would decode tokens and check expiration
        # For now, this is a placeholder
        pass

# Global token blacklist instance
token_blacklist = TokenBlacklist()

def get_current_user_with_blacklist_check(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    Get current customer with blacklist verification
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        Customer ID as integer
        
    Raises:
        HTTPException: If token is blacklisted or authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # Check if token is blacklisted
    if token_blacklist.is_blacklisted(token):
        logger.warning("Attempt to use blacklisted token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return jwt_validator.get_customer_id_from_token(token) 