"""
Proper shared authentication module for all microservices
This replaces the temporary local implementations
"""

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
import jwt
import os
from typing import Dict, Optional, Set
import datetime

# JWT settings from environment
PRIVATE_KEY = os.getenv("JWT_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----")  # Should be loaded securely
PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nYOUR_PUBLIC_KEY_HERE\n-----END PUBLIC KEY-----")  # Should be loaded securely
ALGORITHM = "RS256"

security = HTTPBearer()

# Simple in-memory token blacklist for revocation (replace with Redis in production)
token_blacklist: Set[str] = set()

class JWTService:
    @staticmethod
    def create_token(data: Dict, expires_delta: Optional[int] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_delta)
        else:
            expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, PRIVATE_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> Dict:
        try:
            payload = jwt.decode(token, PUBLIC_KEY, algorithms=[ALGORITHM])
            if token in token_blacklist:
                raise jwt.InvalidTokenError("Token has been revoked")
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    @staticmethod
    def revoke_token(token: str):
        token_blacklist.add(token)

def get_current_customer_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    token = credentials.credentials
    payload = JWTService.decode_token(token)
    customer_id_str = payload.get("sub")
    if customer_id_str is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: missing subject")
    try:
        return int(customer_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid customer ID in token")

def get_current_customer_dict(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Get current customer as a dictionary with additional info
    """
    token = credentials.credentials
    payload = JWTService.decode_token(token)
    customer_id_str = payload.get("sub")
    if customer_id_str is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: missing subject")
    try:
        customer_id = int(customer_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid customer ID in token")
    return {
        "customer_id": customer_id,
        "email": payload.get("email"),
        "is_verified": payload.get("is_verified", False),
        "is_active": payload.get("is_active", True),
        "is_admin": payload.get("is_admin", False)
    }

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
    return JWTService.decode_token(token)

# For revocation, e.g., on logout
# JWTService.revoke_token(some_token)