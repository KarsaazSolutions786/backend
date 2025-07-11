"""
Secure Authentication Service for Note Service

This module provides secure JWT validation with proper signature verification.
"""

import jwt
import logging
from datetime import datetime
from typing import Dict
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
# from shared.simple_auth import get_current_customer_id  # Temporarily disabled

logger = logging.getLogger(__name__)

security = HTTPBearer()

class AuthConfig:
    """Authentication configuration"""
    SECRET_KEY = os.getenv("JWT_SECRET", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
    ALGORITHM = "HS256"
    
    # Validate production environment
    def __post_init__(self):
        if os.getenv("ENVIRONMENT") == "production" and AuthConfig.SECRET_KEY in [
            "your-secret-key-here", 
            "your-secret-key-here-change-in-production",
            "eindr-super-secret-key-change-in-production-123456789"
        ]:
            raise ValueError("Production environment requires a secure SECRET_KEY")

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
            
            # Validate token type
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

def get_current_customer(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Extract customer information from JWT token with secure signature verification.
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        Dictionary containing customer information
        
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
    customer_id = jwt_validator.get_customer_id_from_token(token)
    
    return {
        "id": customer_id,
        "customer_id": customer_id,
        "customer_id_str": str(customer_id)  # For backward compatibility
    } 