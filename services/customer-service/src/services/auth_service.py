from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)
security = HTTPBearer()

class AuthService:
    """Service for handling authentication with auth-service"""
    
    def __init__(self):
        self.auth_service_url = "http://auth-service:8000"  # Internal Docker network
    
    async def verify_token(self, token: str) -> Dict:
        """Verify token with auth service"""
        try:
            response = requests.get(
                f"{self.auth_service_url}/auth/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to verify token with auth service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable"
            )

# Initialize auth service
auth_service = AuthService()

async def get_current_customer(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current customer from auth token"""
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required"
        )
    
    token = credentials.credentials
    customer_data = await auth_service.verify_token(token)
    
    return {
        "customer_id": int(customer_data.get("id")),  # Ensure integer type for DB queries
        "email": customer_data.get("email"),
        "is_verified": customer_data.get("is_verified", False)
    } 