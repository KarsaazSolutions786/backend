from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
import hashlib
import secrets
import uuid
import logging
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import bcrypt
from typing import Dict, Optional

from ..models import Customer, CustomerSession, LoginAttempt
from ..config import settings
from ..database import get_db

logger = logging.getLogger(__name__)

security = HTTPBearer()

class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def hash_token(self, token: str) -> str:
        """Hash a token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def create_customer(self, email: str, password: str) -> Customer:
        """Create a new customer"""
        # Check if customer already exists
        existing_customer = self.db.query(Customer).filter(Customer.email == email).first()
        if existing_customer:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Customer with this email already exists"
            )
        
        # Hash password and create customer
        password_hash = self.hash_password(password)
        customer = Customer(
            email=email,
            password_hash=password_hash,
            is_verified=False,
            is_active=True
        )
        
        self.db.add(customer)
        self.db.commit()
        self.db.refresh(customer)
        
        return customer
    
    def authenticate_customer(self, email: str, password: str) -> Optional[Customer]:
        """Authenticate a customer"""
        customer = self.db.query(Customer).filter(Customer.email == email).first()
        
        if not customer:
            return None
        
        if not customer.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        # Check if account is locked
        if customer.locked_until and customer.locked_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked until {customer.locked_until}"
            )
        
        # Verify password
        if not self.verify_password(password, customer.password_hash):
            # Increment failed attempts
            customer.login_attempts += 1
            
            # Lock account after 5 failed attempts
            if customer.login_attempts >= 5:
                customer.locked_until = datetime.utcnow() + timedelta(minutes=30)
            
            self.db.commit()
            return None
            
        # Reset failed attempts on successful login
        customer.login_attempts = 0
        customer.locked_until = None
        customer.last_login = datetime.utcnow()
        self.db.commit()
        
        return customer
    
    def create_session(self, customer_id: int, ip_address: str = None, user_agent: str = None, remember_me: bool = False) -> CustomerSession:
        """Create a new customer session"""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=30 if remember_me else 1)
        
        session = CustomerSession(
            customer_id=customer_id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
            
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        return session
    
    def log_login_attempt(self, customer_id: int, email: str, success: bool, ip_address: str = None, user_agent: str = None, failure_reason: str = None):
        """Log a login attempt"""
        attempt = LoginAttempt(
            customer_id=customer_id,
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            is_success=success,
            failure_reason=failure_reason
        )
        
        self.db.add(attempt)
        self.db.commit()
    
    def get_customer_by_id(self, customer_id: int) -> Optional[Customer]:
        """Get customer by ID"""
        return self.db.query(Customer).filter(Customer.id == customer_id).first()
    
    def invalidate_sessions(self, customer_id: int):
        """Invalidate all sessions for a customer"""
        self.db.query(CustomerSession).filter(CustomerSession.customer_id == customer_id).delete()
        self.db.commit()

class JWTService:
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(self, data: dict) -> str:
        """Create an access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: dict) -> str:
        """Create a refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode a token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def get_customer_id_from_token(self, token: str) -> int:
        """Extract customer ID from token"""
        payload = self.verify_token(token)
        customer_id = payload.get("sub")
        
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        return int(customer_id)

# Dependency functions for FastAPI
async def get_current_customer(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Dict:
    """Get current authenticated customer"""
    jwt_service = JWTService()
    
    try:
        # Verify token
        payload = jwt_service.verify_token(credentials.credentials)
        customer_id = payload.get("sub")
        
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get customer from database
        auth_service = AuthService(db)
        customer = auth_service.get_customer_by_id(int(customer_id))
        
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
        
        # Return customer data as dict for easy access
        return {
            "customer_id": customer.id,
            "email": customer.email,
            "is_verified": customer.is_verified,
            "is_active": customer.is_active
        }
        
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_optional_current_customer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[Dict]:
    """Get current customer if token is provided, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_customer(credentials, db)
    except HTTPException:
        return None

def require_admin(current_customer: Dict = Depends(get_current_customer)):
    """Require admin access"""
    if not current_customer.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_customer

def require_verified(current_customer: Dict = Depends(get_current_customer)):
    """Require verified account"""
    if not current_customer.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account verification required"
        )
    return current_customer

def create_service_auth_header(customer_id: int) -> dict:
    """Create auth header for internal service-to-service communication"""
    jwt_service = JWTService()
    token = jwt_service.create_access_token({"sub": str(customer_id)})
    return {"Authorization": f"Bearer {token}"}

def verify_service_token(token: str) -> int:
    """Verify token from internal service-to-service communication"""
    jwt_service = JWTService()
    return jwt_service.get_customer_id_from_token(token) 