from sqlalchemy.orm import Session, joinedload
from fastapi import HTTPException, status, Request
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging
import traceback

from models import Customer, CustomerSession, LoginAttempt, CustomerProfile
from schemas import (
    RegisterRequest, CustomerLogin, TokenResponse, 
    CustomerWithProfileResponse, CustomerProfileResponse
)
from config import settings
from services.auth_service import AuthService
from services.jwt_service import JWTService
from utils.shared_importer import get_shared_classes

logger = logging.getLogger(__name__)

# Initialize shared classes
RefreshTokenBase, RefreshTokenService = get_shared_classes()

# Check if enhanced security is available
try:
    from rate_limiting import brute_force_protection, record_auth_failure
    from input_validation import SecureUserRegistration, SecureUserLogin, InputSanitizer, SecurityError
    from security_config import SensitiveDataFilter
    HAS_ENHANCED_SECURITY = True
except ImportError:
    HAS_ENHANCED_SECURITY = False
    logger.warning("Enhanced security modules not available, using basic security")

class AuthBusinessService:
    """Business logic service for authentication operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.auth_service = AuthService(db)
        self.jwt_service = JWTService()
        self.refresh_token_service = self._get_refresh_token_service()
    
    def _get_refresh_token_service(self):
        """Get refresh token service with Redis if available"""
        redis_client = None
        if settings.REDIS_URL and settings.REDIS_URL.strip():
            try:
                import redis
                redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                redis_client.ping()
                logger.info("Redis connection established for RefreshTokenService")
            except Exception as e:
                logger.warning(f"Redis not available for RefreshTokenService: {e}")
                redis_client = None
        else:
            logger.info("Redis not configured, RefreshTokenService will use database-only mode")
        
        return RefreshTokenService(self.db, redis_client)
    
    def validate_request_security(self, request: Request, endpoint: str):
        """Validate request for security (rate limiting, etc.)"""
        if HAS_ENHANCED_SECURITY:
            try:
                brute_force_protection.check_rate_limit(request, endpoint)
            except HTTPException as e:
                logger.warning(f"Rate limit exceeded for {endpoint}: {e.detail}")
                raise
    
    def _validate_registration_data(self, customer_data: RegisterRequest) -> Tuple[str, str]:
        """Validate registration data with enhanced security if available"""
        if HAS_ENHANCED_SECURITY:
            try:
                secure_data = SecureUserRegistration(
                    email=customer_data.email,
                    password=customer_data.password
                )
                return secure_data.email, secure_data.password
            except (ValueError, SecurityError) as e:
                logger.warning(f"Invalid registration data: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid registration data. Please check your input."
                )
        else:
            return customer_data.email, customer_data.password
    
    def _validate_login_data(self, login_data: CustomerLogin, request: Request) -> Tuple[str, str, bool]:
        """Validate login data with enhanced security if available"""
        if HAS_ENHANCED_SECURITY:
            try:
                secure_data = SecureUserLogin(
                    email=login_data.email,
                    password=login_data.password,
                    remember_me=getattr(login_data, 'remember_me', False)
                )
                return secure_data.email, secure_data.password, secure_data.remember_me
            except (ValueError, SecurityError) as e:
                logger.warning(f"Invalid login data: {e}")
                if HAS_ENHANCED_SECURITY:
                    record_auth_failure(request, "auth.login")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid login data"
                )
        else:
            return login_data.email, login_data.password, getattr(login_data, 'remember_me', False)
    
    def _build_customer_response(self, customer: Customer) -> CustomerWithProfileResponse:
        """Build customer response with profile information"""
        try:
            # Get customer with profile information
            customer_with_profile = self.db.query(Customer).options(
                joinedload(Customer.profile)
            ).filter(Customer.id == customer.id).first()
            
            if not customer_with_profile:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Customer not found"
                )
            
            # Build response
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
                profile=CustomerProfileResponse(
                    full_name=getattr(customer_with_profile.profile, 'full_name', None) if customer_with_profile.profile else None,
                    gender=getattr(customer_with_profile.profile, 'gender', None) if customer_with_profile.profile else None,
                    is_new=getattr(customer_with_profile.profile, 'is_new', True) if customer_with_profile.profile else True
                ) if customer_with_profile.profile else None
            )
        except Exception as e:
            logger.error(f"Error constructing customer response: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return minimal response on error
            return CustomerWithProfileResponse(
                id=customer.id,
                email=customer.email,
                is_verified=customer.is_verified,
                is_active=customer.is_active,
                created_at=customer.created_at,
                last_login=customer.last_login,
                login_attempts=0,
                locked_until=None,
                subscription_plan_id=None,
                profile=None
            )
    
    def _create_tokens(self, customer: Customer, request: Request, session_id: Optional[int] = None) -> Tuple[str, str]:
        """Create access and refresh tokens"""
        # Generate JWT tokens
        token_data = {
            "sub": str(customer.id),
            "email": customer.email,
            "iat": datetime.utcnow(),
            "permissions": ["user"]
        }
        
        if session_id:
            token_data["session_id"] = session_id
        
        access_token = self.jwt_service.create_access_token(data=token_data)
        
        # Create refresh token
        refresh_token, _ = self.refresh_token_service.create_refresh_token(
            customer_id=customer.id,
            device_id=request.headers.get("x-device-id"),
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host
        )
        
        return access_token, refresh_token
    
    def register_customer(self, customer_data: RegisterRequest, request: Request) -> TokenResponse:
        """Register a new customer"""
        try:
            # Validate password confirmation
            if customer_data.password != customer_data.confirm_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Passwords do not match"
                )
            
            # Apply security validation
            self.validate_request_security(request, "auth.register")
            
            # Validate input data
            validated_email, validated_password = self._validate_registration_data(customer_data)
            
            # Create customer
            customer = self.auth_service.create_customer(
                validated_email,
                validated_password,
                customer_data.full_name,
                customer_data.gender,
                customer_data.is_new
            )
            
            # Log successful registration
            self.auth_service.log_login_attempt(
                customer_id=customer.id,
                email=validated_email,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
                success=True
            )
            
            # Create session
            session = self.auth_service.create_session(
                customer_id=customer.id,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
                remember_me=False
            )
            
            # Create tokens
            access_token, refresh_token = self._create_tokens(customer, request)
            
            # Build response
            customer_response = self._build_customer_response(customer)
            
            logger.info(f"Customer registered successfully: {validated_email}")
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                customer=customer_response
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Registration error: {type(e).__name__}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed. Please try again later."
            )
    
    def login_customer(self, login_data: CustomerLogin, request: Request) -> TokenResponse:
        """Authenticate and login a customer"""
        try:
            # Apply security validation
            self.validate_request_security(request, "auth.login")
            
            # Validate input data
            validated_email, validated_password, remember_me = self._validate_login_data(login_data, request)
            
            # Authenticate customer
            customer = self.auth_service.authenticate_customer(validated_email, validated_password)
            
            if not customer:
                # Log failed attempt
                self.auth_service.log_login_attempt(
                    customer_id=None,
                    email=validated_email,
                    ip_address=request.client.host,
                    user_agent=request.headers.get("user-agent"),
                    success=False,
                    failure_reason="invalid_credentials"
                )
                
                # Record failed attempt for brute force protection
                if HAS_ENHANCED_SECURITY:
                    record_auth_failure(request, "auth.login")
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Log successful login
            self.auth_service.log_login_attempt(
                customer_id=customer.id,
                email=validated_email,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
                success=True
            )
            
            # Create session
            session = self.auth_service.create_session(
                customer_id=customer.id,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
                remember_me=remember_me
            )
            
            # Create tokens
            access_token, refresh_token = self._create_tokens(customer, request, session.id)
            
            # Build response
            customer_response = self._build_customer_response(customer)
            
            logger.info(f"Customer logged in successfully: {validated_email}")
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                customer=customer_response
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Login error: {type(e).__name__}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Login temporarily unavailable. Please try again later."
            )
    
    def refresh_token(self, refresh_token: str, request: Request) -> TokenResponse:
        """Refresh access token"""
        try:
            # Apply rate limiting
            self.validate_request_security(request, "auth.token_refresh")
            
            # Validate refresh token
            token_record = self.refresh_token_service.validate_token(refresh_token)
            if not token_record:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired refresh token"
                )
            
            # Get customer
            customer = self.auth_service.get_customer_by_id(token_record.customer_id)
            if not customer or not customer.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Customer not found or inactive"
                )
            
            # Use token rotation for enhanced security
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
            
            # Build response
            customer_response = self._build_customer_response(customer)
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                customer=customer_response
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {type(e).__name__}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token refresh failed"
            )
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate access token and return customer info"""
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
                "expires_at": payload.get("exp")
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {type(e).__name__}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token validation failed"
            )
    
    def logout_customer(self, customer_id: int, refresh_token: str) -> Dict[str, str]:
        """Logout customer and revoke tokens"""
        try:
            # Revoke all refresh tokens for the customer
            revoked_count = self.refresh_token_service.revoke_all_tokens(customer_id, "logout")
            logger.info(f"Revoked {revoked_count} refresh tokens for customer {customer_id}")
            
            # Invalidate all sessions for the customer
            self.auth_service.invalidate_sessions(customer_id)
            
            logger.info(f"Customer logged out: {customer_id}")
            
            return {"message": "Successfully logged out"}
            
        except Exception as e:
            logger.error(f"Logout error: {type(e).__name__}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed"
            )