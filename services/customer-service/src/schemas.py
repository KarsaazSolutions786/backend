from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"

class SubscriptionType(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class DeviceType(str, Enum):
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"

# Customer Profile Schemas
class CustomerProfileBase(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=500)
    phone_number: Optional[str] = Field(None, max_length=20)
    timezone: Optional[str] = "UTC"
    language: Optional[str] = "en"
    is_public: Optional[bool] = False

class CustomerProfileCreate(CustomerProfileBase):
    customer_id: int
    
class CustomerProfileUpdate(CustomerProfileBase):
    pass

class CustomerProfileResponse(CustomerProfileBase):
    id: int
    customer_id: int
    avatar_url: Optional[str] = None
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime
    is_new: bool = True  # Required boolean field
    
    class Config:
        from_attributes = True

# Customer Schemas
class CustomerBase(BaseModel):
    email: Optional[str] = None
    is_active: Optional[bool] = True
    is_verified: Optional[bool] = False
    
class CustomerCreate(CustomerBase):
    email: str
    password_hash: str
    
class CustomerUpdate(BaseModel):
    # Customer fields
    email: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_new: Optional[bool] = None
    
    # Profile-related fields
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=500)
    phone_number: Optional[str] = Field(None, max_length=20)
    timezone: Optional[str] = "UTC"
    language: Optional[str] = "en"
    is_public: Optional[bool] = False

class CustomerResponse(CustomerBase):
    id: int
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    profile: Optional[CustomerProfileResponse] = None  # Ensure profile is included
    
    class Config:
        from_attributes = True

# Customer Preferences Schemas
class CustomerPreferencesBase(BaseModel):
    email_notifications: Optional[bool] = True
    push_notifications: Optional[bool] = True
    sms_notifications: Optional[bool] = False
    default_reminder_time: Optional[str] = Field("09:00", pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    reminder_sound: Optional[str] = "default"
    snooze_duration: Optional[int] = Field(5, ge=1, le=60)
    profile_visibility: Optional[str] = Field("private", pattern="^(public|friends|private)$")
    activity_visibility: Optional[str] = Field("private", pattern="^(public|friends|private)$")
    theme: Optional[str] = Field("light", pattern="^(light|dark|auto)$")
    date_format: Optional[str] = "YYYY-MM-DD"
    time_format: Optional[str] = Field("24h", pattern="^(12h|24h)$")
    ai_suggestions: Optional[bool] = True
    location_services: Optional[bool] = False
    analytics_tracking: Optional[bool] = True

class CustomerPreferencesCreate(CustomerPreferencesBase):
    customer_id: int
    
class CustomerPreferencesUpdate(CustomerPreferencesBase):
    pass

class CustomerPreferencesResponse(CustomerPreferencesBase):
    id: int
    customer_id: int
    created_at: Optional[datetime] = None
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Customer Device Schemas
class CustomerDeviceBase(BaseModel):
    device_token: str
    device_type: DeviceType
    device_name: Optional[str] = None
    supports_push: Optional[bool] = True
    supports_location: Optional[bool] = False
    is_active: Optional[bool] = True

class CustomerDeviceCreate(CustomerDeviceBase):
    pass

class CustomerDeviceUpdate(BaseModel):
    device_name: Optional[str] = None
    supports_push: Optional[bool] = None
    supports_location: Optional[bool] = None
    is_active: Optional[bool] = None

class CustomerDeviceResponse(CustomerDeviceBase):
    id: int
    customer_id: int
    created_at: datetime
    updated_at: datetime
    last_used: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# List Response Schemas
class CustomersListResponse(BaseModel):
    customers: List[CustomerResponse]
    total: int
    page: int
    limit: int
    pages: int

# Stats Schemas
class CustomerStats(BaseModel):
    total_customers: int
    active_customers: int
    verified_customers: int
    inactive_customers: int 