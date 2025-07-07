from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List
from datetime import datetime
import re

# Request Schemas
class CustomerRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    
    @validator('password')
    def validate_password(cls, v):
        if not re.search(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]", v):
            raise ValueError('Password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character')
        return v

class CustomerLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: Optional[bool] = False

class TokenRefresh(BaseModel):
    refresh_token: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_password(cls, v):
        if not re.search(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]", v):
            raise ValueError('Password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character')
        return v

class EmailVerification(BaseModel):
    token: str

# Response Schemas
class CustomerBase(BaseModel):
    id: int
    email: str
    is_verified: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True

class CustomerResponse(CustomerBase):
    login_attempts: int
    locked_until: Optional[datetime]
    subscription_plan_id: Optional[int]

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    customer: CustomerBase

class CustomerSessionResponse(BaseModel):
    id: int
    customer_id: int
    ip_address: Optional[str]
    user_agent: Optional[str]
    expires_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class LoginAttemptResponse(BaseModel):
    id: int
    customer_id: int
    email: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_success: bool
    failure_reason: Optional[str]
    attempted_at: datetime
    
    class Config:
        from_attributes = True

class CustomerWithSessions(CustomerResponse):
    sessions: List[CustomerSessionResponse] = []
    recent_login_attempts: List[LoginAttemptResponse] = []

# Error Schemas
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

class ValidationErrorResponse(BaseModel):
    detail: str
    errors: List[dict]

# Update Schemas
class CustomerUpdate(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    subscription_plan_id: Optional[int] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_password(cls, v):
        if not re.search(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]", v):
            raise ValueError('Password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character')
        return v 