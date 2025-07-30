from sqlalchemy.orm import Session, joinedload
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import traceback

from ..models import Customer, CustomerProfile
from ..schemas import CustomerWithProfileResponse, CustomerProfileResponse, TokenResponse
from ..config import settings

logger = logging.getLogger(__name__)

class ResponseService:
    """Service for handling response formatting and data building"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def build_customer_response(self, customer: Customer) -> CustomerWithProfileResponse:
        """Build customer response with profile information"""
        try:
            # Get customer with profile information using joinedload for efficiency
            customer_with_profile = self.db.query(Customer).options(
                joinedload(Customer.profile)
            ).filter(Customer.id == customer.id).first()
            
            if not customer_with_profile:
                logger.error(f"Customer not found when building response: {customer.id}")
                # Fallback to provided customer object
                customer_with_profile = customer
            
            # Build profile response
            profile_response = None
            if hasattr(customer_with_profile, 'profile') and customer_with_profile.profile:
                profile_response = CustomerProfileResponse(
                    full_name=getattr(customer_with_profile.profile, 'full_name', None),
                    gender=getattr(customer_with_profile.profile, 'gender', None),
                    is_new=getattr(customer_with_profile.profile, 'is_new', True)
                )
            
            # Build main customer response
            return CustomerWithProfileResponse(
                id=customer_with_profile.id,
                email=customer_with_profile.email,
                is_verified=customer_with_profile.is_verified,
                is_active=customer_with_profile.is_active,
                created_at=customer_with_profile.created_at,
                last_login=customer_with_profile.last_login,
                login_attempts=getattr(customer_with_profile, 'login_attempts', 0),
                locked_until=getattr(customer_with_profile, 'locked_until', None),
                subscription_plan_id=getattr(customer_with_profile, 'subscription_plan_id', None),
                profile=profile_response
            )
            
        except Exception as e:
            logger.error(f"Error building customer response: {type(e).__name__}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return minimal safe response on error
            return self._build_minimal_customer_response(customer)
    
    def _build_minimal_customer_response(self, customer: Customer) -> CustomerWithProfileResponse:
        """Build minimal customer response when full response fails"""
        try:
            return CustomerWithProfileResponse(
                id=customer.id,
                email=customer.email,
                is_verified=getattr(customer, 'is_verified', False),
                is_active=getattr(customer, 'is_active', True),
                created_at=getattr(customer, 'created_at', None),
                last_login=getattr(customer, 'last_login', None),
                login_attempts=0,
                locked_until=None,
                subscription_plan_id=None,
                profile=None
            )
        except Exception as e:
            logger.error(f"Error building minimal customer response: {e}")
            # Last resort - create response with just ID and email
            return CustomerWithProfileResponse(
                id=customer.id,
                email=customer.email,
                is_verified=False,
                is_active=True,
                created_at=None,
                last_login=None,
                login_attempts=0,
                locked_until=None,
                subscription_plan_id=None,
                profile=None
            )
    
    def build_token_response(
        self, 
        access_token: str, 
        refresh_token: str, 
        customer: Customer,
        expires_in: Optional[int] = None
    ) -> TokenResponse:
        """Build complete token response"""
        try:
            # Use provided expires_in or default from settings
            token_expires_in = expires_in or (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
            
            # Build customer response
            customer_response = self.build_customer_response(customer)
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=token_expires_in,
                customer=customer_response
            )
            
        except Exception as e:
            logger.error(f"Error building token response: {type(e).__name__}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Build minimal token response
            minimal_customer = self._build_minimal_customer_response(customer)
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=token_expires_in,
                customer=minimal_customer
            )
    
    def build_customer_profile_response(self, customer_id: int) -> Optional[CustomerProfileResponse]:
        """Build customer profile response"""
        try:
            # Get customer profile
            profile = self.db.query(CustomerProfile).filter(
                CustomerProfile.customer_id == customer_id
            ).first()
            
            if not profile:
                return None
            
            return CustomerProfileResponse(
                full_name=profile.full_name,
                gender=profile.gender,
                is_new=profile.is_new
            )
            
        except Exception as e:
            logger.error(f"Error building profile response: {type(e).__name__}: {str(e)}")
            return None
    
    def build_validation_response(self, customer: Customer, token_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build token validation response"""
        try:
            return {
                "valid": True,
                "customer_id": customer.id,
                "email": customer.email,
                "is_verified": customer.is_verified,
                "is_active": customer.is_active,
                "expires_at": token_payload.get("exp"),
                "issued_at": token_payload.get("iat"),
                "permissions": token_payload.get("permissions", []),
                "session_id": token_payload.get("session_id")
            }
            
        except Exception as e:
            logger.error(f"Error building validation response: {type(e).__name__}: {str(e)}")
            # Return minimal validation response
            return {
                "valid": True,
                "customer_id": customer.id,
                "email": customer.email,
                "is_verified": getattr(customer, 'is_verified', False),
                "is_active": getattr(customer, 'is_active', True),
                "expires_at": token_payload.get("exp"),
                "issued_at": token_payload.get("iat"),
                "permissions": [],
                "session_id": None
            }
    
    def build_error_response(self, error_code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build standardized error response"""
        response = {
            "error": {
                "code": error_code,
                "message": message
            }
        }
        
        if details:
            response["error"]["details"] = details
        
        return response
    
    def build_success_response(self, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build standardized success response"""
        response = {
            "success": True,
            "message": message
        }
        
        if data:
            response["data"] = data
        
        return response
    
    def build_logout_response(self, revoked_tokens_count: int = 0) -> Dict[str, str]:
        """Build logout response"""
        message = "Successfully logged out"
        if revoked_tokens_count > 0:
            message += f" and revoked {revoked_tokens_count} tokens"
        
        return {"message": message}
    
    def build_health_response(self) -> Dict[str, Any]:
        """Build health check response"""
        return {
            "status": "healthy",
            "service": "auth-service",
            "timestamp": str(datetime.utcnow()),
            "version": getattr(settings, 'SERVICE_VERSION', '1.0.0')
        }
    
    def sanitize_customer_data_for_logs(self, customer: Customer) -> Dict[str, Any]:
        """Sanitize customer data for safe logging"""
        return {
            "customer_id": customer.id,
            "email": customer.email[:3] + "***" + customer.email.split('@')[1] if '@' in customer.email else "***",
            "is_verified": customer.is_verified,
            "is_active": customer.is_active,
            "created_at": str(customer.created_at) if customer.created_at else None
        }
    
    def format_customer_for_token(self, customer: Customer) -> Dict[str, Any]:
        """Format customer data for inclusion in JWT tokens"""
        return {
            "id": customer.id,
            "email": customer.email,
            "is_verified": customer.is_verified,
            "is_active": customer.is_active
        }