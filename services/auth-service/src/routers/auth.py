from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from typing import List, Optional
from datetime import datetime, timedelta
import bcrypt
import jwt
import secrets
import logging

from ..database import get_db
from ..models import Customer, CustomerSession, LoginAttempt
from ..schemas import (
    CustomerRegister, CustomerLogin, TokenResponse, CustomerResponse,
    CustomerWithSessions, LoginAttemptResponse, CustomerUpdate,
    PasswordChange, PasswordResetRequest, PasswordReset, ErrorResponse,
    CustomerSessionResponse, TokenRefresh
)
from ..services.auth_service import AuthService, JWTService, get_current_customer
from ..config import settings

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

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_customer(
    customer_data: CustomerRegister,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new customer"""
    try:
        # Create customer using auth service
        customer = auth_service.create_customer(customer_data.email, customer_data.password)
        
        # Log successful registration attempt
        auth_service.log_login_attempt(
            customer_id=customer.id,
            email=customer_data.email,
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
        
        # Generate JWT tokens
        access_token = jwt_service.create_access_token(data={"sub": str(customer.id)})
        refresh_token = jwt_service.create_refresh_token(data={"sub": str(customer.id)})
        
        logger.info(f"Customer registered successfully: {customer_data.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            customer=CustomerResponse.from_orm(customer)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=TokenResponse)
async def login_customer(
    login_data: CustomerLogin,
    request: Request,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Authenticate customer and return tokens"""
    try:
        # Authenticate customer
        customer = auth_service.authenticate_customer(login_data.email, login_data.password)
        
        if not customer:
            # Log failed attempt
            auth_service.log_login_attempt(
                customer_id=None,
                email=login_data.email,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent"),
                success=False,
                failure_reason="invalid_credentials"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Log successful login
        auth_service.log_login_attempt(
            customer_id=customer.id,
            email=login_data.email,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            success=True
        )
        
        # Create session
        session = auth_service.create_session(
            customer_id=customer.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            remember_me=getattr(login_data, 'remember_me', False)
        )
        
        # Generate JWT tokens
        access_token = jwt_service.create_access_token(data={"sub": str(customer.id)})
        refresh_token = jwt_service.create_refresh_token(data={"sub": str(customer.id)})
        
        logger.info(f"Customer logged in successfully: {login_data.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            customer=CustomerResponse.from_orm(customer)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    token_data: TokenRefresh,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = jwt_service.verify_token(token_data.refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        customer_id = payload.get("sub")
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get customer
        customer = auth_service.get_customer_by_id(int(customer_id))
        if not customer or not customer.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Customer not found or inactive"
            )
        
        # Generate new tokens
        access_token = jwt_service.create_access_token(data={"sub": str(customer.id)})
        new_refresh_token = jwt_service.create_refresh_token(data={"sub": str(customer.id)})
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            customer=CustomerResponse.from_orm(customer)
        )
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.get("/me", response_model=CustomerWithSessions)
async def get_current_customer_info(
    current_customer: dict = Depends(get_current_customer),
    db: Session = Depends(get_db)
):
    """Get current customer information with sessions"""
    try:
        customer_id = current_customer["customer_id"]
        
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
    current_customer: dict = Depends(get_current_customer),
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout customer and invalidate all sessions"""
    try:
        customer_id = current_customer["customer_id"]
        auth_service.invalidate_sessions(customer_id)
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.post("/validate-token")
async def validate_token(
    current_customer: dict = Depends(get_current_customer)
):
    """Validate token and return customer info"""
    return current_customer 