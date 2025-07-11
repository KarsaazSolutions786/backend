import jwt
import httpx
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ..config import settings

logger = logging.getLogger(__name__)

class AuthService:
    """Authentication service for AI pipeline service-to-service communication"""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self._cached_token = None
        self._token_expiry = None
    
    def create_service_token(self, customer_id: str) -> str:
        """Create a JWT token for service-to-service communication"""
        try:
            to_encode = {
                "sub": str(customer_id),
                "type": "service",
                "service": "ai-pipeline",
                "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
                "iat": datetime.utcnow()
            }
            
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"Service token created for customer: {customer_id}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Error creating service token: {e}")
            raise
    
    async def get_auth_token_from_auth_service(self, customer_id: str) -> Optional[str]:
        """Get a JWT token from the auth service for a specific customer"""
        try:
            # First, try to login with service account to get a token
            async with httpx.AsyncClient(timeout=settings.TIMEOUT) as client:
                login_data = {
                    "email": settings.SERVICE_ACCOUNT_EMAIL,
                    "password": settings.SERVICE_ACCOUNT_PASSWORD
                }
                
                resp = await client.post(f"{settings.AUTH_URL}/auth/login", json=login_data)
                if resp.status_code == 200:
                    token_data = resp.json()
                    return token_data.get("access_token")
                else:
                    logger.warning(f"Failed to get auth token from auth service: {resp.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting auth token from auth service: {e}")
            return None
    
    async def get_valid_token(self, customer_id: str) -> Optional[str]:
        """Get a valid token, either from cache or by creating a new one"""
        try:
            # Check if we have a cached token that's still valid
            if self._cached_token and self._token_expiry and datetime.utcnow() < self._token_expiry:
                return self._cached_token
            
            # Try to get token from auth service first
            auth_token = await self.get_auth_token_from_auth_service(customer_id)
            if auth_token:
                self._cached_token = auth_token
                self._token_expiry = datetime.utcnow() + timedelta(minutes=25)  # Cache for 25 minutes
                return auth_token
            
            # Fallback to creating a service token
            service_token = self.create_service_token(customer_id)
            self._cached_token = service_token
            self._token_expiry = datetime.utcnow() + timedelta(minutes=25)
            return service_token
            
        except Exception as e:
            logger.error(f"Error getting valid token: {e}")
            return None
    
    def get_auth_headers(self, token: str) -> Dict[str, str]:
        """Get authentication headers for HTTP requests"""
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        } 