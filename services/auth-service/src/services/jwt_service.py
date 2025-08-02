import jwt
from jwt.exceptions import InvalidTokenError as JWTError
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import traceback

from ..config import settings

logger = logging.getLogger(__name__)

class JWTService:
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create an access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "iss": "eindr-issuer",  # Required by Kong JWT plugin
            "aud": "eindr-api"      # Required by Kong JWT plugin
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.debug(f"Access token created for user: {data.get('sub')}")
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a refresh token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "iss": "eindr-issuer",  # Required by Kong JWT plugin
            "aud": "eindr-api"      # Required by Kong JWT plugin
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.debug(f"Refresh token created for user: {data.get('sub')}")
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode a token"""
        try:
            # Log token details for debugging
            logger.info(f"Verifying token: {token[:10]}...")
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Validate token payload
            if not payload:
                logger.warning("Token payload is empty")
                raise jwt.JWTError("Empty token payload")
            
            # Check required fields
            required_fields = ["sub", "email", "type", "exp"]
            for field in required_fields:
                if field not in payload:
                    logger.warning(f"Missing required field: {field}")
                    raise jwt.JWTError(f"Missing required field: {field}")
            
            # Validate token type
            if payload.get("type") not in ["access", "refresh"]:
                logger.warning(f"Invalid token type: {payload.get('type')}")
                raise jwt.JWTError("Invalid token type")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise
        
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token error: {str(e)}")
            raise
        
        except Exception as e:
            logger.error(f"Unexpected token verification error: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise jwt.JWTError("Could not validate token")
    
    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a refresh token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "refresh":
                raise JWTError("Invalid token type")
            
            return payload
        
        except JWTError as e:
            logger.warning(f"Refresh token verification failed: {e}")
            raise JWTError("Could not validate refresh token")
    
    def get_token_payload(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token payload without verification (for debugging)"""
        try:
            # Decode without verification
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        
        except Exception as e:
            logger.error(f"Error decoding token payload: {e}")
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """Check if a token is expired"""
        try:
            payload = self.get_token_payload(token)
            if not payload:
                return True
            
            exp = payload.get("exp")
            if not exp:
                return True
            
            return datetime.fromtimestamp(exp) < datetime.utcnow()
        
        except Exception:
            return True
    
    def get_remaining_time(self, token: str) -> Optional[timedelta]:
        """Get remaining time before token expires"""
        try:
            payload = self.get_token_payload(token)
            if not payload:
                return None
            
            exp = payload.get("exp")
            if not exp:
                return None
            
            expire_time = datetime.fromtimestamp(exp)
            remaining = expire_time - datetime.utcnow()
            
            return remaining if remaining.total_seconds() > 0 else None
        
        except Exception:
            return None