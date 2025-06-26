from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
import logging

from ..database import get_db
from ..models import User, UserPreferences, UserDevice
from ..services.user_service import UserService

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class UserProfileCreate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=500)
    phone_number: Optional[str] = Field(None, max_length=20)
    timezone: str = Field(default="UTC")
    language: str = Field(default="en")
    is_public: bool = Field(default=False)

class UserProfileUpdate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=150)
    bio: Optional[str] = Field(None, max_length=500)
    phone_number: Optional[str] = Field(None, max_length=20)
    timezone: Optional[str] = None
    language: Optional[str] = None
    is_public: Optional[bool] = None

class UserProfileResponse(BaseModel):
    id: str
    auth_user_id: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    phone_number: Optional[str]
    timezone: str
    language: str
    is_public: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

class UserPreferencesUpdate(BaseModel):
    email_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    sms_notifications: Optional[bool] = None
    default_reminder_time: Optional[str] = Field(None, regex=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    reminder_sound: Optional[str] = None
    snooze_duration: Optional[int] = Field(None, ge=1, le=60)
    profile_visibility: Optional[str] = Field(None, regex="^(public|friends|private)$")
    activity_visibility: Optional[str] = Field(None, regex="^(public|friends|private)$")
    theme: Optional[str] = Field(None, regex="^(light|dark|auto)$")
    date_format: Optional[str] = None
    time_format: Optional[str] = Field(None, regex="^(12h|24h)$")
    ai_suggestions: Optional[bool] = None
    location_services: Optional[bool] = None
    analytics_tracking: Optional[bool] = None

class UserPreferencesResponse(BaseModel):
    id: str
    user_id: str
    email_notifications: bool
    push_notifications: bool
    sms_notifications: bool
    default_reminder_time: str
    reminder_sound: str
    snooze_duration: int
    profile_visibility: str
    activity_visibility: str
    theme: str
    date_format: str
    time_format: str
    ai_suggestions: bool
    location_services: bool
    analytics_tracking: bool
    created_at: datetime
    updated_at: datetime

class DeviceRegistration(BaseModel):
    device_token: str
    device_type: str = Field(..., regex="^(ios|android|web)$")
    device_name: Optional[str] = None
    supports_push: bool = True
    supports_location: bool = False

# Initialize service
user_service = UserService()

async def get_current_user(request: Request):
    """Get current user from auth token"""
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ")[1]
    # TODO: Validate token with auth service
    # For now, mock user validation
    return {"user_id": "mock-user-id", "email": "user@example.com"}

@router.post("/profile", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(
    profile_data: UserProfileCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create user profile"""
    try:
        profile = await user_service.create_profile(
            db=db,
            auth_user_id=current_user["user_id"],
            profile_data=profile_data.dict()
        )
        
        logger.info(f"Profile created for user: {current_user['user_id']}")
        
        return UserProfileResponse(
            id=str(profile.id),
            auth_user_id=profile.auth_user_id,
            first_name=profile.first_name,
            last_name=profile.last_name,
            display_name=profile.display_name,
            bio=profile.bio,
            avatar_url=profile.avatar_url,
            phone_number=profile.phone_number,
            timezone=profile.timezone,
            language=profile.language,
            is_public=profile.is_public,
            is_verified=profile.is_verified,
            created_at=profile.created_at,
            updated_at=profile.updated_at
        )
    
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create profile"
        )

@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get user profile"""
    try:
        profile = await user_service.get_profile(
            db=db,
            auth_user_id=current_user["user_id"]
        )
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        return UserProfileResponse(
            id=str(profile.id),
            auth_user_id=profile.auth_user_id,
            first_name=profile.first_name,
            last_name=profile.last_name,
            display_name=profile.display_name,
            bio=profile.bio,
            avatar_url=profile.avatar_url,
            phone_number=profile.phone_number,
            timezone=profile.timezone,
            language=profile.language,
            is_public=profile.is_public,
            is_verified=profile.is_verified,
            created_at=profile.created_at,
            updated_at=profile.updated_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile"
        )

@router.put("/profile", response_model=UserProfileResponse)
async def update_profile(
    profile_data: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    try:
        profile = await user_service.update_profile(
            db=db,
            auth_user_id=current_user["user_id"],
            update_data=profile_data.dict(exclude_unset=True)
        )
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        logger.info(f"Profile updated for user: {current_user['user_id']}")
        
        return UserProfileResponse(
            id=str(profile.id),
            auth_user_id=profile.auth_user_id,
            first_name=profile.first_name,
            last_name=profile.last_name,
            display_name=profile.display_name,
            bio=profile.bio,
            avatar_url=profile.avatar_url,
            phone_number=profile.phone_number,
            timezone=profile.timezone,
            language=profile.language,
            is_public=profile.is_public,
            is_verified=profile.is_verified,
            created_at=profile.created_at,
            updated_at=profile.updated_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.post("/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Upload user avatar"""
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only JPEG, PNG, and GIF are allowed."
            )
        
        # Validate file size (5MB max)
        if file.size > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 5MB."
            )
        
        avatar_url = await user_service.upload_avatar(
            db=db,
            auth_user_id=current_user["user_id"],
            file=file
        )
        
        logger.info(f"Avatar uploaded for user: {current_user['user_id']}")
        
        return {"avatar_url": avatar_url}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading avatar: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload avatar"
        )

@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_preferences(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get user preferences"""
    try:
        preferences = await user_service.get_preferences(
            db=db,
            user_id=current_user["user_id"]
        )
        
        if not preferences:
            # Create default preferences if none exist
            preferences = await user_service.create_default_preferences(
                db=db,
                user_id=current_user["user_id"]
            )
        
        return UserPreferencesResponse(
            id=str(preferences.id),
            user_id=preferences.user_id,
            email_notifications=preferences.email_notifications,
            push_notifications=preferences.push_notifications,
            sms_notifications=preferences.sms_notifications,
            default_reminder_time=preferences.default_reminder_time,
            reminder_sound=preferences.reminder_sound,
            snooze_duration=preferences.snooze_duration,
            profile_visibility=preferences.profile_visibility,
            activity_visibility=preferences.activity_visibility,
            theme=preferences.theme,
            date_format=preferences.date_format,
            time_format=preferences.time_format,
            ai_suggestions=preferences.ai_suggestions,
            location_services=preferences.location_services,
            analytics_tracking=preferences.analytics_tracking,
            created_at=preferences.created_at,
            updated_at=preferences.updated_at
        )
    
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get preferences"
        )

@router.put("/preferences", response_model=UserPreferencesResponse)
async def update_preferences(
    preferences_data: UserPreferencesUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update user preferences"""
    try:
        preferences = await user_service.update_preferences(
            db=db,
            user_id=current_user["user_id"],
            update_data=preferences_data.dict(exclude_unset=True)
        )
        
        if not preferences:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preferences not found"
            )
        
        logger.info(f"Preferences updated for user: {current_user['user_id']}")
        
        return UserPreferencesResponse(
            id=str(preferences.id),
            user_id=preferences.user_id,
            email_notifications=preferences.email_notifications,
            push_notifications=preferences.push_notifications,
            sms_notifications=preferences.sms_notifications,
            default_reminder_time=preferences.default_reminder_time,
            reminder_sound=preferences.reminder_sound,
            snooze_duration=preferences.snooze_duration,
            profile_visibility=preferences.profile_visibility,
            activity_visibility=preferences.activity_visibility,
            theme=preferences.theme,
            date_format=preferences.date_format,
            time_format=preferences.time_format,
            ai_suggestions=preferences.ai_suggestions,
            location_services=preferences.location_services,
            analytics_tracking=preferences.analytics_tracking,
            created_at=preferences.created_at,
            updated_at=preferences.updated_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

@router.post("/devices")
async def register_device(
    device_data: DeviceRegistration,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Register a device for notifications"""
    try:
        device = await user_service.register_device(
            db=db,
            user_id=current_user["user_id"],
            device_data=device_data.dict()
        )
        
        logger.info(f"Device registered for user: {current_user['user_id']}")
        
        return {
            "id": str(device.id),
            "message": "Device registered successfully"
        }
    
    except Exception as e:
        logger.error(f"Error registering device: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register device"
        )

@router.get("/devices")
async def get_devices(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get user's registered devices"""
    try:
        devices = await user_service.get_user_devices(
            db=db,
            user_id=current_user["user_id"]
        )
        
        return [
            {
                "id": str(device.id),
                "device_type": device.device_type,
                "device_name": device.device_name,
                "supports_push": device.supports_push,
                "supports_location": device.supports_location,
                "is_active": device.is_active,
                "last_used_at": device.last_used_at,
                "created_at": device.created_at
            }
            for device in devices
        ]
    
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get devices"
        )

@router.delete("/devices/{device_id}")
async def unregister_device(
    device_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Unregister a device"""
    try:
        success = await user_service.unregister_device(
            db=db,
            user_id=current_user["user_id"],
            device_id=device_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        
        logger.info(f"Device unregistered: {device_id} for user: {current_user['user_id']}")
        
        return {"message": "Device unregistered successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering device: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unregister device"
        ) 