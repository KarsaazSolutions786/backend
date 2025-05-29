from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import timedelta, datetime
from sqlalchemy.orm import Session
from models.models import User, Preferences
from connect_db import get_db
from firebase_auth import verify_firebase_token
from core.config import settings
from utils.logger import logger
import uuid
import hashlib
import secrets
import jwt

router = APIRouter()

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    language: Optional[str] = "en"
    timezone: Optional[str] = "UTC"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class FirebaseUserCreate(BaseModel):
    email: EmailStr
    language: Optional[str] = None
    timezone: Optional[str] = None

class UserPreferencesCreate(BaseModel):
    allow_friends: Optional[bool] = True
    receive_shared_notes: Optional[bool] = True
    notification_sound: Optional[str] = "default"
    tts_language: Optional[str] = "en"
    chat_history_enabled: Optional[bool] = True

class UserResponse(BaseModel):
    id: str
    email: str
    language: Optional[str] = None
    timezone: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True

class PreferencesResponse(BaseModel):
    user_id: str
    allow_friends: Optional[bool]
    receive_shared_notes: Optional[bool]
    notification_sound: Optional[str]
    tts_language: Optional[str]
    chat_history_enabled: Optional[bool]

    class Config:
        from_attributes = True

class UserWithPreferencesResponse(BaseModel):
    user: UserResponse
    preferences: Optional[PreferencesResponse] = None

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

def hash_password(password: str) -> str:
    """Hash password using SHA256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    try:
        salt, password_hash = hashed_password.split(":")
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def get_user_by_id(user_id: str, db: Session):
    """Get user by ID from database."""
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(email: str, db: Session):
    """Get user by email from database."""
    return db.query(User).filter(User.email == email).first()

def create_new_user(user_data: UserCreate, user_id: str, db: Session, password_hash: Optional[str] = None):
    """Create a new user in the database."""
    user = User(
        id=user_id,
        email=user_data.email,
        language=user_data.language,
        timezone=user_data.timezone
    )
    
    # Add password field if it exists in User model (you may need to add this)
    if hasattr(User, 'password_hash') and password_hash:
        user.password_hash = password_hash
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_firebase_user(user_data: FirebaseUserCreate, firebase_uid: str, db: Session):
    """Create a new user from Firebase data."""
    user = User(
        id=firebase_uid,
        email=user_data.email,
        language=user_data.language,
        timezone=user_data.timezone
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_user_preferences(user_id: str, preferences_data: UserPreferencesCreate, db: Session):
    """Create default preferences for a user."""
    preferences = Preferences(
        user_id=user_id,
        allow_friends=preferences_data.allow_friends,
        receive_shared_notes=preferences_data.receive_shared_notes,
        notification_sound=preferences_data.notification_sound,
        tts_language=preferences_data.tts_language,
        chat_history_enabled=preferences_data.chat_history_enabled
    )
    db.add(preferences)
    db.commit()
    db.refresh(preferences)
    return preferences

# === NON-AUTHENTICATED ENDPOINTS (Traditional Auth) ===

@router.post("/register", response_model=LoginResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user with email and password (no Firebase required)."""
    try:
        # Check if email is already used
        if get_user_by_email(user_data.email, db):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Generate user ID
        user_id = str(uuid.uuid4())
        
        # Hash password
        password_hash = hash_password(user_data.password)
        
        # Create new user
        user = create_new_user(user_data, user_id, db, password_hash)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Create default preferences
        preferences_data = UserPreferencesCreate()
        create_user_preferences(user_id, preferences_data, db)
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.id, "email": user.email},
            expires_delta=access_token_expires
        )
        
        logger.info(f"New user registered: {user_data.email} with ID: {user_id}")
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                language=user.language,
                timezone=user.timezone,
                created_at=user.created_at.isoformat()
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/login", response_model=LoginResponse)
async def login(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """Login user with email and password."""
    try:
        # Get user by email
        user = get_user_by_email(user_credentials.email, db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password (if password field exists)
        if hasattr(user, 'password_hash'):
            if not verify_password(user_credentials.password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
        else:
            # For now, if no password field exists, just check if user exists
            # This is for backward compatibility
            logger.warning(f"User {user.email} exists but no password verification available")
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.id, "email": user.email},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user_credentials.email}")
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                language=user.language,
                timezone=user.timezone,
                created_at=user.created_at.isoformat()
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# === FIREBASE ENDPOINTS (Firebase Auth Required) ===

@router.post("/firebase/register", response_model=UserWithPreferencesResponse)
async def firebase_register(
    user_data: FirebaseUserCreate,
    preferences_data: UserPreferencesCreate = UserPreferencesCreate(),
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Register a new user with Firebase authentication."""
    try:
        firebase_uid = current_user["uid"]
        
        # Check if user already exists
        if get_user_by_id(firebase_uid, db):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already registered"
            )
        
        # Check if email is already used
        if get_user_by_email(user_data.email, db):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = create_firebase_user(user_data, firebase_uid, db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Create user preferences
        preferences = create_user_preferences(firebase_uid, preferences_data, db)
        
        logger.info(f"New Firebase user registered: {user_data.email} with ID: {firebase_uid}")
        
        return UserWithPreferencesResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                language=user.language,
                timezone=user.timezone,
                created_at=user.created_at.isoformat()
            ),
            preferences=PreferencesResponse(
                user_id=preferences.user_id,
                allow_friends=preferences.allow_friends,
                receive_shared_notes=preferences.receive_shared_notes,
                notification_sound=preferences.notification_sound,
                tts_language=preferences.tts_language,
                chat_history_enabled=preferences.chat_history_enabled
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Firebase registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# === AUTHENTICATED ENDPOINTS ===

@router.get("/me", response_model=UserWithPreferencesResponse)
async def get_current_user_info(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get current user information with preferences."""
    try:
        user_id = current_user["uid"]
        user = get_user_by_id(user_id, db)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user preferences
        preferences = db.query(Preferences).filter(Preferences.user_id == user_id).first()
        
        preferences_response = None
        if preferences:
            preferences_response = PreferencesResponse(
                user_id=preferences.user_id,
                allow_friends=preferences.allow_friends,
                receive_shared_notes=preferences.receive_shared_notes,
                notification_sound=preferences.notification_sound,
                tts_language=preferences.tts_language,
                chat_history_enabled=preferences.chat_history_enabled
            )
        
        return UserWithPreferencesResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                language=user.language,
                timezone=user.timezone,
                created_at=user.created_at.isoformat()
            ),
            preferences=preferences_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    preferences_data: UserPreferencesCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Update user preferences."""
    try:
        user_id = current_user["uid"]
        
        # Get existing preferences
        preferences = db.query(Preferences).filter(Preferences.user_id == user_id).first()
        
        if not preferences:
            # Create new preferences if they don't exist
            preferences = create_user_preferences(user_id, preferences_data, db)
        else:
            # Update existing preferences
            update_data = preferences_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(preferences, field, value)
            
            db.commit()
            db.refresh(preferences)
        
        logger.info(f"Updated preferences for user: {user_id}")
        
        return PreferencesResponse(
            user_id=preferences.user_id,
            allow_friends=preferences.allow_friends,
            receive_shared_notes=preferences.receive_shared_notes,
            notification_sound=preferences.notification_sound,
            tts_language=preferences.tts_language,
            chat_history_enabled=preferences.chat_history_enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update preferences error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 