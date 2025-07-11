"""
Simplified Authentication for Eindr Microservices

This module provides easy-to-use authentication functions that services can
import and use directly, replacing their individual JWT utilities.
"""

from typing import Dict, Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth_utils import jwt_validator

# Security bearer for dependency injection
security = HTTPBearer()

def get_current_customer_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    Get current authenticated customer ID - direct replacement for old jwt.py functions
    
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

def get_current_user_data(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Get current authenticated customer's full data
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        User data dictionary with customer_id, email, etc.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = jwt_validator.verify_token(token, "access")
    customer_id = jwt_validator.get_customer_id_from_token(token)
    
    return {
        "customer_id": customer_id,
        "email": payload.get("email", ""),
        "is_verified": payload.get("is_verified", True),
        "is_active": payload.get("is_active", True)
    }

def get_optional_customer_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[int]:
    """
    Get current customer ID if authenticated, otherwise return None
    
    Args:
        credentials: Optional JWT credentials
        
    Returns:
        Customer ID or None if not authenticated
    """
    if not credentials:
        return None
    
    try:
        return get_current_customer_id(credentials)
    except HTTPException:
        return None

# Legacy compatibility - for services still using old patterns
get_current_user = get_current_customer_id  # Alias for backward compatibility 