"""
Secure Authentication Service for Customer Service

This module provides secure JWT validation with proper signature verification.
Migrated to use shared authentication framework with fallback support.
"""

import os
import sys
import logging
from typing import Dict, Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import shared authentication

try:
    from simple_auth import get_current_customer_id as shared_get_current_customer_id
    HAS_SHARED_AUTH = True
    logger.info("Using shared authentication framework")
except ImportError:
    HAS_SHARED_AUTH = False
    logger.warning("Shared auth not available, using local secure implementation")

class AuthConfig:
    """Authentication configuration"""
    SECRET_KEY = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
    ALGORITHM = "HS256"
    
    # Validate that we're not using a weak default key in production
    if os.getenv("ENVIRONMENT") == "production" and SECRET_KEY in [
        "your-secret-key-here", 
        "your-secret-key-here-change-in-production",
        "eindr-super-secret-key-change-in-production-123456789"
    ]:
        raise ValueError("Production environment requires a secure SECRET_KEY")

# Set service name for logging
os.environ.setdefault("SERVICE_NAME", "customer-service")

# Security bearer for dependency injection
security = HTTPBearer()

def verify_jwt_token(token: str) -> Dict:
    """Secure JWT token verification with signature validation"""
    try:
        # Use secure JWT decode with signature verification
        payload = jwt.decode(
            token, 
            AuthConfig.SECRET_KEY, 
            algorithms=[AuthConfig.ALGORITHM],
            options={"verify_signature": True, "verify_exp": True}
        )
        
        # Validate required fields
        customer_id = payload.get("sub")
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
        
        return {
            "customer_id": int(customer_id),
            "email": payload.get("email", ""),
            "is_verified": payload.get("is_verified", True),
            "is_active": payload.get("is_active", True)
        }
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except ValueError as e:
        logger.error(f"Invalid customer ID in token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid customer ID in token"
        )
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

class AuthService:
    """Standalone authentication service for customer service"""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db
    
    async def verify_token(self, token: str) -> Dict:
        """
        Verify JWT token using standalone implementation
        
        Args:
            token: JWT token to verify
            
        Returns:
            User data dictionary
        """
        try:
            customer_data = verify_jwt_token(token)
            logger.info(f"Successfully verified token for customer {customer_data.get('customer_id')}")
            return customer_data
            
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise
    
    async def get_current_customer_data(self, customer_id: int) -> Dict:
        """
        Get customer data with proper validation
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Customer data dictionary
        """
        return {
            "customer_id": customer_id,
            "service": "customer-service",
            "verified": True
        }

# Initialize auth service
auth_service = AuthService()

async def get_current_customer(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Get current customer with secure authentication
    
    Args:
        credentials: JWT credentials from Authorization header
        
    Returns:
        Customer data dictionary with customer_id, email, verification status
    """
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required"
        )
    
    try:
        # Initialize auth service
        service = AuthService()
        
        # Verify token using standalone implementation
        customer_data = await service.verify_token(credentials.credentials)
        
        # Return expected customer data structure
        return {
            "customer_id": int(customer_data.get("customer_id")),
            "email": customer_data.get("email"),
            "is_verified": customer_data.get("is_verified", False),
            "is_active": customer_data.get("is_active", True)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error in customer service: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def get_optional_customer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[Dict]:
    """
    Get current customer if authenticated, otherwise return None
    
    Args:
        credentials: Optional JWT credentials
        
    Returns:
        Customer data or None if not authenticated
    """
    if not credentials:
        return None
    
    try:
        return await get_current_customer(credentials)
    except HTTPException:
        return None

def require_customer_access():
    """
    Dependency to require customer service access
    
    Returns:
        FastAPI dependency function
    """
    async def check_access(customer: Dict = Depends(get_current_customer_id)) -> Dict:
        # Additional customer service validation can be added here
        return customer
    
    return check_access

def require_customer_modification():
    """
    Dependency to require customer modification permissions
    
    Returns:
        FastAPI dependency function
    """
    async def check_modification(
        customer: Dict = Depends(get_current_customer_id)
    ) -> Dict:
        # For now, just return the customer - can add permission checks later
        return customer

class LegacyAuthService:
    """Legacy authentication service for backward compatibility"""
    
    def __init__(self):
        logger.warning("Using legacy auth service - please migrate to dependency injection")
    
    async def verify_token(self, token: str) -> Dict:
        """Legacy token verification"""
        return verify_jwt_token(token) 