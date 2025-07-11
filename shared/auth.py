"""
Proper shared authentication module for all microservices
This replaces the temporary local implementations
"""

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
import jwt
import os
from typing import Dict, Optional

# JWT settings from environment
SECRET_KEY = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
ALGORITHM = "HS256"

security = HTTPBearer()

def get_current_customer_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    Extract customer ID from JWT token
    Used by all microservices for authentication
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Get current customer as a dictionary with additional info
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[int]:
    """
    Get current customer ID if token is provided, otherwise return None
    """
    if not credentials:
        return None
        
    try:
        return get_current_customer_id(credentials)
    except HTTPException:
        return None

def verify_token(token: str) -> Dict:
    """
    Verify a token and return payload (for internal service communication)
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
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

def create_token(data: Dict, expires_delta: Optional[int] = None) -> str:
    """
    Create a JWT token (for auth service use)
    """
    import datetime
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_delta)
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt 