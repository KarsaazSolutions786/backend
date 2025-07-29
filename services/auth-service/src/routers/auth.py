from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, desc
from typing import List, Optional
from datetime import datetime, timedelta
import bcrypt
import jwt
import secrets
import logging
import sys
import os
import traceback

# Import RefreshTokenService from shared module
import sys
import os

# Add the backend directory to Python path to access shared module
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from shared.refresh_token_service import RefreshTokenService
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported RefreshTokenService from shared module")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import RefreshTokenService from shared module: {e}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Backend path: {backend_path}")
    logger.error(f"Python path: {sys.path}")
    # Define a minimal RefreshTokenService class to prevent NameError
    class RefreshTokenService:
        def __init__(self, db, redis_client=None):
            self.db = db
            self.redis_client = redis_client
            logger.warning("Using minimal RefreshTokenService implementation")
        
        def create_refresh_token(self, customer_id, device_id=None, user_agent=None, ip_address=None):
            return secrets.token_urlsafe(64), None
            
        def validate_token(self, token):
            return None
            
        def revoke_token(self, token, reason="manual"):
            return True
            
        def revoke_all_tokens(self, customer_id):
            return True

from src.database import get_db
from src.models import Customer, CustomerSession, LoginAttempt
from src.schemas import (
    CustomerRegister, CustomerLogin, TokenResponse, CustomerResponse,
    CustomerWithSessions, LoginAttemptResponse, CustomerUpdate,
    PasswordChange, PasswordResetRequest, PasswordReset, ErrorResponse,
    CustomerSessionResponse, TokenRefresh, RegisterRequest,
    CustomerWithProfileResponse, CustomerProfileResponse
)
from src.config import settings
from src.services.auth_service import AuthService
from src.services.jwt_service import JWTService
# from shared.simple_auth import get_current_customer_id  # Temporarily disabled

# Temporary local implementation
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
import jwt
from src.config import settings

def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
    """Temporary local implementation of get_current_customer_id"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        customer_id = payload.get("sub")
        if customer_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return int(customer_id)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Import enhanced security modules
try:
    from rate_limiting import brute_force_protection, record_auth_failure
    from input_validation import SecureUserRegistration, SecureUserLogin, InputSanitizer, SecurityError
    from security_config import SensitiveDataFilter
    HAS_ENHANCED_SECURITY = True
except ImportError:
    print("Warning: Enhanced security modules not available, using basic security")
    HAS_ENHANCED_SECURITY = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
security = HTTPBearer()

# Initialize JWT service
jwt_service = JWTService()

def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    """Get AuthService instance with database dependency"""
    return AuthService(db)

def get_refresh_token_service(db: Session = Depends(get_db)):
    """Get refresh token service instance"""
    # Initialize Redis client if available and configured
    redis_client = None
    if settings.REDIS_URL and settings.REDIS_URL.strip():
        try:
            import redis
            redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            # Test connection with timeout
            redis_client.ping()
            logger.info("Redis connection established for RefreshTokenService")
        except Exception as e:
            logger.warning(f"Redis not available for RefreshTokenService: {e}")
            redis_client = None
    else:
        logger.info("Redis not configured, RefreshTokenService will use database-only mode")
    
    return RefreshTokenService(db, redis_client)

# Token revocation functions removed - now handled directly by RefreshTokenService

def validate_request_security(request: Request, endpoint: str):
    """Validate request for security (rate limiting, etc.)"""
    if HAS_ENHANCED_SECURITY:
        try:
            brute_force_protection.check_rate_limit(request, endpoint)
        except HTTPException as e:
            logger.warning(f"Rate limit exceeded for {endpoint}: {e.detail}")
            raise

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_customer(
    customer_data: RegisterRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    refresh_token_service = Depends(get_refresh_token_service)
):
    """Register a new customer with enhanced security"""
    
    # Validate password confirmation
    if customer_data.password != customer_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Apply rate limiting and security checks
    validate_request_security(request, "auth.register")
    
    try:
        # Enhanced input validation
        if HAS_ENHANCED_SECURITY:
            # Use secure validation model
            try:
                secure_data = SecureUserRegistration(
                    email=customer_data.email,
                    password=customer_data.password
                )
                validated_email = secure_data.email
                validated_password = secure_data.password
            except (ValueError, SecurityError) as e:
                logger.warning(f"Invalid registration data: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid registration data. Please check your input."
                )
        else:
            validated_email = customer_data.email
            validated_password = customer_data.password
        
        # Create customer using auth service
        customer = auth_service.create_customer(
            validated_email, 
            validated_password, 
            customer_data.full_name, 
            customer_data.gender,
            customer_data.is_new
        )
        
        # Log successful registration attempt (with sanitized data)
        log_data = {"email": validated_email, "success": True}
        if HAS_ENHANCED_SECURITY:
            log_data = SensitiveDataFilter.filter_sensitive_data(log_data)
        
        auth_service.log_login_attempt(
            customer_id=customer.id,
            email=validated_email,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            success=True
        )
        
        # Create session
        session = auth_service.create_session(
            customer_id=customer.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            remember_me=False
        )
        
        # Generate JWT tokens with enhanced security
        token_data = {
            "sub": str(customer.id),
            "email": validated_email,
            "iat": datetime.utcnow(),
            "permissions": ["user"]  # Basic customer permissions
        }
        
        access_token = jwt_service.create_access_token(data=token_data)
        
        # Create refresh token using RefreshTokenService
        refresh_token, _ = refresh_token_service.create_refresh_token(
            customer_id=customer.id,
            device_id=request.headers.get("x-device-id"),
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host
        )
        
        logger.info(f"Customer registered successfully: {validated_email}")
        
        # Get customer with profile information
        customer_with_profile = db.query(Customer).options(
            joinedload(Customer.profile)
        ).filter(Customer.id == customer.id).first()
        
        # Manually construct the response to handle the profile relationship
        try:
            customer_response = CustomerWithProfileResponse(
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
            customer_response = CustomerWithProfileResponse(
                id=customer_with_profile.id,
                email=customer_with_profile.email,
                is_verified=customer_with_profile.is_verified,
                is_active=customer_with_profile.is_active,
                created_at=customer_with_profile.created_at,
                last_login=customer_with_profile.last_login,
                login_attempts=0,
                locked_until=None,
                subscription_plan_id=None,
                profile=None
            )
        
        response = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            customer=customer_response
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {type(e).__name__}: {str(e)}")  # Log full error details
        logger.error(f"Registration traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration error: {type(e).__name__}: {str(e)}"
        )

@router.post("/login", response_model=TokenResponse)
async def login_customer(
    login_data: CustomerLogin,
    request: Request,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    refresh_token_service = Depends(get_refresh_token_service)
):
    """Authenticate customer with enhanced security"""
    
    # Apply rate limiting and security checks
    validate_request_security(request, "auth.login")
    
    try:
        # Enhanced input validation
        if HAS_ENHANCED_SECURITY:
            try:
                secure_data = SecureUserLogin(
                    email=login_data.email,
                    password=login_data.password,
                    remember_me=getattr(login_data, 'remember_me', False)
                )
                validated_email = secure_data.email
                validated_password = secure_data.password
                remember_me = secure_data.remember_me
            except (ValueError, SecurityError) as e:
                logger.warning(f"Invalid login data: {e}")
                # Record failed attempt for brute force protection
                if HAS_ENHANCED_SECURITY:
                    record_auth_failure(request, "auth.login")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid login data"
                )
        else:
            validated_email = login_data.email
            validated_password = login_data.password
            remember_me = getattr(login_data, 'remember_me', False)
        
        # Authenticate customer
        customer = auth_service.authenticate_customer(validated_email, validated_password)
        
        if not customer:
            # Log failed attempt
            auth_service.log_login_attempt(
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
        auth_service.log_login_attempt(
            customer_id=customer.id,
            email=validated_email,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            success=True
        )
        
        # Create session
        session = auth_service.create_session(
            customer_id=customer.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            remember_me=remember_me
        )
        
        # Generate JWT tokens with enhanced security
        token_data = {
            "sub": str(customer.id),
            "email": validated_email,
            "iat": datetime.utcnow(),
            "permissions": ["user"],  # Basic customer permissions
            "session_id": session.id  # Link to session for revocation
        }
        
        access_token = jwt_service.create_access_token(data=token_data)
        
        # Create refresh token using RefreshTokenService
        refresh_token, _ = refresh_token_service.create_refresh_token(
            customer_id=customer.id,
            device_id=request.headers.get("x-device-id"),
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host
        )
        
        logger.info(f"Customer logged in successfully: {validated_email}")
        
        # Get customer with profile information
        customer_with_profile = db.query(Customer).options(
            joinedload(Customer.profile)
        ).filter(Customer.id == customer.id).first()
        
        # Manually construct the response to handle the profile relationship
        try:
            customer_response = CustomerWithProfileResponse(
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
            customer_response = CustomerWithProfileResponse(
                id=customer_with_profile.id,
                email=customer_with_profile.email,
                is_verified=customer_with_profile.is_verified,
                is_active=customer_with_profile.is_active,
                created_at=customer_with_profile.created_at,
                last_login=customer_with_profile.last_login,
                login_attempts=0,
                locked_until=None,
                subscription_plan_id=None,
                profile=None
            )
        
        response = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            customer=customer_response
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {type(e).__name__}")  # Don't log sensitive details
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login temporarily unavailable. Please try again later."
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefresh,
    request: Request,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    refresh_token_service = Depends(get_refresh_token_service)
):
    """Refresh access token with enhanced security"""
    
    # Apply rate limiting
    validate_request_security(request, "auth.token_refresh")
    
    try:
        # Validate refresh token using RefreshTokenService
        token_record = refresh_token_service.validate_token(token_data.refresh_token)
        if not token_record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Get customer
        customer = auth_service.get_customer_by_id(token_record.customer_id)
        if not customer or not customer.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Customer not found or inactive"
            )
        
        # Use token rotation for enhanced security
        new_refresh_token, new_token_record = refresh_token_service.rotate_token(
            old_token=token_data.refresh_token,
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
        
        access_token = jwt_service.create_access_token(data=new_token_data)
        
        # Get customer with profile information
        customer_with_profile = db.query(Customer).options(
            joinedload(Customer.profile)
        ).filter(Customer.id == customer.id).first()
        
        # Check if customer_with_profile is None
        if customer_with_profile is None:
            logger.error(f"Customer with profile not found for ID: {customer.id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Manually construct the response to handle the profile relationship
        try:
            customer_response = CustomerWithProfileResponse(
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
            customer_response = CustomerWithProfileResponse(
                id=customer_with_profile.id,
                email=customer_with_profile.email,
                is_verified=customer_with_profile.is_verified,
                is_active=customer_with_profile.is_active,
                created_at=customer_with_profile.created_at,
                last_login=customer_with_profile.last_login,
                login_attempts=0,
                locked_until=None,
                subscription_plan_id=None,
                profile=None
            )
        
        response = TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            customer=customer_response
        )
        
        return response
    
    except jwt.ExpiredSignatureError:
        logger.error("Refresh token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired"
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

@router.get("/me", response_model=CustomerWithSessions)
async def get_current_customer_info(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get current customer information with sessions"""
    try:
        customer_id = current_customer_id
        
        # Get customer with sessions
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Get sessions
        sessions = db.query(CustomerSession).filter(
            CustomerSession.customer_id == customer_id,
            CustomerSession.expires_at > datetime.utcnow()
        ).all()
        
        # Get recent login attempts
        recent_attempts = db.query(LoginAttempt).filter(
            LoginAttempt.customer_id == customer_id
        ).order_by(desc(LoginAttempt.attempted_at)).limit(10).all()
        
        response = CustomerWithSessions.from_orm(customer)
        response.sessions = [CustomerSessionResponse.from_orm(s) for s in sessions]
        response.recent_login_attempts = [LoginAttemptResponse.from_orm(a) for a in recent_attempts]
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get customer info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get customer info"
        )

@router.post("/logout")
async def logout_customer(
    request: Request,
    token_data: TokenRefresh,  # Expect refresh token in request body
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    refresh_token_service = Depends(get_refresh_token_service)
):
    """Logout customer and revoke refresh token"""
    try:
        # Revoke all refresh tokens for the customer
        customer_id = current_customer_id
        
        if customer_id:
            # Revoke all tokens for the customer
            revoked_count = refresh_token_service.revoke_all_tokens(customer_id, "logout")
            logger.info(f"Revoked {revoked_count} refresh tokens for customer {customer_id}")
            
            # Invalidate all sessions for the customer
            auth_service.invalidate_sessions(customer_id)
        
        logger.info(f"Customer logged out: {customer_id}")
        
        response = {"message": "Successfully logged out"}
        return response
    
    except Exception as e:
        logger.error(f"Logout error: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.post("/revoke-token")
async def revoke_token_endpoint(
    token_data: TokenRefresh,
    current_customer_id: int = Depends(get_current_customer_id),
    refresh_token_service: RefreshTokenService = Depends(get_refresh_token_service)
):
    """Revoke a specific refresh token"""
    try:
        # Validate the token and check ownership
        token_record = refresh_token_service.validate_token(token_data.refresh_token)
        if not token_record:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired token"
            )
        
        if token_record.customer_id != current_customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot revoke token belonging to another user"
            )
        
        # Revoke the token
        success = refresh_token_service.revoke_token(token_data.refresh_token, "manual")
        if not success:
            logger.warning(f"Failed to revoke token for customer {current_customer_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to revoke token"
            )
        
        return {"message": "Token revoked successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token revocation error: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token revocation failed"
        )

@router.post("/validate-token")
async def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    refresh_token_service = Depends(get_refresh_token_service)
):
    """Validate token and return customer info"""
    try:
        if not credentials:
            logger.warning("No token provided in validate_token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No token provided"
            )
        
        token = credentials.credentials
        logger.info(f"Validating token: {token[:10]}...")  # Log first 10 chars of token
        
        # Note: Access tokens are not stored in RefreshTokenService
        # They are stateless JWT tokens that expire naturally
        
        # Verify token
        try:
            payload = jwt_service.verify_token(token)
            logger.info(f"Token payload: {payload}")
        except Exception as verify_error:
            logger.error(f"Token verification error: {type(verify_error).__name__} - {str(verify_error)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        
        response = {
            "valid": True,
            "customer_id": payload.get("sub"),
            "email": payload.get("email"),
            "permissions": payload.get("permissions", []),
            "expires_at": payload.get("exp")
        }
        
        return response
    
    except HTTPException:
        raise
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Token validation error: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token validation failed"
        )