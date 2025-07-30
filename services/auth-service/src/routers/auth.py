import traceback
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import Customer
from src.schemas import (
    CustomerCreate, CustomerLogin, TokenResponse, CustomerResponse,
    CustomerWithProfileResponse, TokenRefresh, TokenValidation, 
    TokenValidationResponse
)
from src.services.auth_business_service import AuthBusinessService
from src.services.token_service import TokenService
from src.services.security_service import SecurityService
from src.services.response_service import ResponseService
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security
security = HTTPBearer()

# Dependencies
def get_auth_business_service(db: Session = Depends(get_db)) -> AuthBusinessService:
    """Get AuthBusinessService instance"""
    return AuthBusinessService(db)

def get_token_service(db: Session = Depends(get_db)) -> TokenService:
    """Get TokenService instance"""
    return TokenService(db)

def get_security_service() -> SecurityService:
    """Get SecurityService instance"""
    return SecurityService()

def get_response_service(db: Session = Depends(get_db)) -> ResponseService:
    """Get ResponseService instance"""
    return ResponseService(db)

def get_current_customer_id(
    credentials: HTTPAuthorizationCredentials = Depends(security), 
    token_service: TokenService = Depends(get_token_service)
) -> int:
    """Extract customer ID from JWT token"""
    try:
        token = credentials.credentials
        validation_result = token_service.validate_access_token(token)
        return validation_result["customer_id"]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    customer_data: CustomerCreate,
    request: Request,
    auth_business_service: AuthBusinessService = Depends(get_auth_business_service)
):
    """Register a new customer"""
    return auth_business_service.register_customer(customer_data, request)

@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: CustomerLogin,
    request: Request,
    auth_business_service: AuthBusinessService = Depends(get_auth_business_service)
):
    """Authenticate customer and return tokens"""
    return auth_business_service.login_customer(login_data, request)

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: dict,
    request: Request,
    auth_business_service: AuthBusinessService = Depends(get_auth_business_service)
):
    """Refresh access token using refresh token"""
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token is required"
        )
    
    return auth_business_service.refresh_token(refresh_token, request)

@router.get("/me", response_model=CustomerWithProfileResponse)
async def get_current_customer_info(
    current_customer_id: int = Depends(get_current_customer_id),
    response_service: ResponseService = Depends(get_response_service),
    db: Session = Depends(get_db)
):
    """Get current customer information"""
    try:
        # Get customer
        customer = db.query(Customer).filter(Customer.id == current_customer_id).first()
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        return response_service.build_customer_response(customer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve customer information"
        )

@router.post("/logout", response_model=dict)
async def logout(
    request: Request,
    current_customer_id: int = Depends(get_current_customer_id),
    auth_business_service: AuthBusinessService = Depends(get_auth_business_service)
):
    """Logout customer and invalidate session"""
    try:
        return auth_business_service.logout_customer(current_customer_id, request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout"
        )



@router.post("/validate-token", response_model=TokenValidationResponse)
async def validate_token(
    request: Request,
    token_data: TokenValidation,
    auth_business_service: AuthBusinessService = Depends(get_auth_business_service)
):
    """Validate access token"""
    try:
        return auth_business_service.validate_token(token_data.token, request)
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return TokenValidationResponse(
            valid=False,
            customer_id=None,
            error="Token validation failed"
        )