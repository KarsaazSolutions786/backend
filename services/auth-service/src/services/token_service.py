from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from fastapi import HTTPException, status, Request
import logging

from .jwt_service import JWTService
from ..utils.shared_importer import get_shared_classes
from ..config import settings
from ..models import Customer

logger = logging.getLogger(__name__)

# Initialize shared classes
RefreshTokenBase, RefreshTokenService = get_shared_classes()

class TokenService:
    """Service for handling all token operations"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.jwt_service = JWTService()
        self.refresh_token_service = self._initialize_refresh_token_service()
    
    def _initialize_refresh_token_service(self):
        """Initialize refresh token service with Redis if available"""
        redis_client = None
        
        if settings.REDIS_URL and settings.REDIS_URL.strip():
            try:
                import redis
                redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                redis_client.ping()
                logger.info("Redis connection established for TokenService")
            except Exception as e:
                logger.warning(f"Redis not available for TokenService: {e}")
                redis_client = None
        else:
            logger.info("Redis not configured, TokenService will use database-only mode")
        
        return RefreshTokenService(self.db, redis_client)
    
    def create_token_pair(self, customer: Customer, request: Request, session_id: Optional[int] = None) -> Tuple[str, str]:
        """Create access and refresh token pair"""
        try:
            # Prepare token data
            token_data = {
                "sub": str(customer.id),
                "email": customer.email,
                "iat": datetime.utcnow(),
                "permissions": ["user"]
            }
            
            if session_id:
                token_data["session_id"] = session_id
            
            # Create access token
            access_token = self.jwt_service.create_access_token(data=token_data)
            
            # Create refresh token
            refresh_token, _ = self.refresh_token_service.create_refresh_token(
                customer_id=customer.id,
                device_id=request.headers.get("x-device-id"),
                user_agent=request.headers.get("user-agent"),
                ip_address=request.client.host
            )
            
            return access_token, refresh_token
            
        except Exception as e:
            logger.error(f"Token creation error: {type(e).__name__}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token creation failed"
            )
    
    def refresh_access_token(self, refresh_token: str, request: Request) -> Tuple[str, str, Customer]:
        """Refresh access token using refresh token"""
        try:
            # Validate refresh token
            token_record = self.refresh_token_service.validate_token(refresh_token)
            if not token_record:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired refresh token"
                )
            
            # Get customer
            customer = self.db.query(Customer).filter(Customer.id == token_record.customer_id).first()
            if not customer or not customer.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Customer not found or inactive"
                )
            
            # Rotate refresh token for security
            new_refresh_token, new_token_record = self.refresh_token_service.rotate_token(
                old_token=refresh_token,
                device_id=request.headers.get("x-device-id"),
                user_agent=request.headers.get("user-agent"),
                ip_address=request.client.host
            )
            
            # Generate new access token
            new_token_data = {
                "sub": str(customer.id),
                "email": customer.email,
                "iat": datetime.utcnow(),
                "permissions": ["user"]
            }
            
            access_token = self.jwt_service.create_access_token(data=new_token_data)
            
            return access_token, new_refresh_token, customer
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {type(e).__name__}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token refresh failed"
            )
    
    def validate_access_token(self, token: str) -> Dict[str, Any]:
        """Validate access token and return payload"""
        try:
            # Verify token
            payload = self.jwt_service.verify_token(token)
            
            # Get customer ID from payload
            customer_id = payload.get("sub")
            if not customer_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            # Get customer from database
            customer = self.db.query(Customer).filter(Customer.id == int(customer_id)).first()
            if not customer:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Customer not found"
                )
            
            if not customer.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated"
                )
            
            return {
                "valid": True,
                "customer_id": int(customer_id),
                "email": customer.email,
                "is_verified": customer.is_verified,
                "is_active": customer.is_active,
                "expires_at": payload.get("exp"),
                "payload": payload
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {type(e).__name__}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token validation failed"
            )
    
    def revoke_customer_tokens(self, customer_id: int, reason: str = "logout") -> int:
        """Revoke all tokens for a customer"""
        try:
            revoked_count = self.refresh_token_service.revoke_all_tokens(customer_id, reason)
            logger.info(f"Revoked {revoked_count} tokens for customer {customer_id} (reason: {reason})")
            return revoked_count
            
        except Exception as e:
            logger.error(f"Token revocation error: {type(e).__name__}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token revocation failed"
            )
    
    def revoke_single_token(self, refresh_token: str, reason: str = "logout") -> bool:
        """Revoke a single refresh token"""
        try:
            success = self.refresh_token_service.revoke_token(refresh_token, reason)
            if success:
                logger.info(f"Revoked refresh token (reason: {reason})")
            else:
                logger.warning(f"Failed to revoke refresh token (reason: {reason})")
            return success
            
        except Exception as e:
            logger.error(f"Single token revocation error: {type(e).__name__}: {str(e)}")
            return False
    
    def get_token_info(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Get information about a refresh token"""
        try:
            token_record = self.refresh_token_service.get_token_info(refresh_token)
            if token_record:
                return {
                    "customer_id": token_record.customer_id,
                    "device_id": getattr(token_record, 'device_id', None),
                    "created_at": getattr(token_record, 'created_at', None),
                    "expires_at": getattr(token_record, 'expires_at', None),
                    "is_active": getattr(token_record, 'is_active', True)
                }
            return None
            
        except Exception as e:
            logger.error(f"Token info retrieval error: {type(e).__name__}: {str(e)}")
            return None
    
    def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens"""
        try:
            cleaned_count = self.refresh_token_service.cleanup_expired_tokens()
            logger.info(f"Cleaned up {cleaned_count} expired tokens")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Token cleanup error: {type(e).__name__}: {str(e)}")
            return 0