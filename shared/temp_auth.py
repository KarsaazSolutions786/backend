"""
Temporary authentication module for microservices
This replaces the shared.simple_auth import until we fix the Docker build context issues.
"""

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
import jwt
import os
from typing import Dict, Optional

# Default JWT settings - services can override these
DEFAULT_SECRET_KEY = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
DEFAULT_ALGORITHM = "HS256"

def get_current_customer_id(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    secret_key: str = DEFAULT_SECRET_KEY,
    algorithm: str = DEFAULT_ALGORITHM
) -> int:
    """
    Temporary local implementation of get_current_customer_id
    Extracts customer ID from JWT token
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        customer_id = payload.get("sub")
        if customer_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
        return int(customer_id)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid customer ID in token"
        )

def get_current_customer_dict(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    secret_key: str = DEFAULT_SECRET_KEY,
    algorithm: str = DEFAULT_ALGORITHM
) -> Dict:
    """
    Get current customer as a dictionary with additional info
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        
        customer_id = payload.get("sub")
        if customer_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
            
        return {
            "customer_id": int(customer_id),
            "email": payload.get("email"),
            "is_verified": payload.get("is_verified", False),
            "is_active": payload.get("is_active", True),
            "is_admin": payload.get("is_admin", False)
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid customer ID in token"
        )

def get_optional_current_customer_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    secret_key: str = DEFAULT_SECRET_KEY,
    algorithm: str = DEFAULT_ALGORITHM
) -> Optional[int]:
    """
    Get current customer ID if token is provided, otherwise return None
    """
    if not credentials:
        return None
        
    try:
        return get_current_customer_id(credentials, secret_key, algorithm)
    except HTTPException:
        return None 