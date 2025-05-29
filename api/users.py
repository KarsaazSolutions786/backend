from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from models.models import User
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger

router = APIRouter()

# Pydantic models for request/response
class UserCreate(BaseModel):
    email: str
    language: Optional[str] = None
    timezone: Optional[str] = None

class UserUpdate(BaseModel):
    language: Optional[str] = None
    timezone: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    language: Optional[str] = None
    timezone: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class UserList(BaseModel):
    total: int
    users: List[UserResponse]
    page: int
    page_size: int
    total_pages: int

    class Config:
        from_attributes = True

@router.post("/register", response_model=UserResponse)
async def register_user(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Register a new user after Firebase authentication."""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.id == current_user["uid"]).first()
        if existing_user:
            logger.info(f"User already exists: {current_user['uid']}")
            return existing_user

        # Create new user using Firebase data
        new_user = User(
            id=current_user["uid"],
            email=current_user["email"],
            language=current_user.get("language", "en"),  # Default to English
            timezone=current_user.get("timezone", "UTC")   # Default to UTC
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"New user registered: {current_user['uid']}")
        return new_user

    except Exception as e:
        db.rollback()
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user: {str(e)}"
        )

@router.get("/profile", response_model=UserResponse)
async def get_profile(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get user profile using Firebase UID."""
    try:
        # Query user by Firebase UID
        user = db.query(User).filter(User.id == current_user["uid"]).first()
        if not user:
            # If user doesn't exist, automatically register them
            return await register_user(current_user=current_user, db=db)
        
        logger.info(f"Profile retrieved for user: {current_user['uid']}")
        return user

    except Exception as e:
        logger.error(f"Error retrieving profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve profile: {str(e)}"
        )

@router.put("/profile", response_model=UserResponse)
async def update_profile(
    update_data: UserUpdate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Update user profile."""
    try:
        # Query user by Firebase UID
        user = db.query(User).filter(User.id == current_user["uid"]).first()
        if not user:
            logger.warning(f"User not found: {current_user['uid']}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found. Please register first."
            )

        # Update fields if provided
        if update_data.language is not None:
            user.language = update_data.language
        if update_data.timezone is not None:
            user.timezone = update_data.timezone

        db.commit()
        db.refresh(user)
        
        logger.info(f"Profile updated for user: {current_user['uid']}")
        return user

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}"
        )

@router.get("/allusers", response_model=UserList)
async def get_all_users(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of users per page"),
    sort_by: str = Query("created_at", description="Sort by field (created_at, email, language, timezone)"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    search: Optional[str] = Query(None, description="Search by email or language or timezone"),
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """
    Get all users with pagination and filtering options.
    Only authenticated users can access this endpoint.
    """
    try:
        # Start with base query
        query = db.query(User)

        # Apply search filter if provided
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (User.email.ilike(search_term)) | 
                (User.language.ilike(search_term)) | 
                (User.timezone.ilike(search_term))
            )

        # Get total count before pagination
        total_users = query.count()

        # Apply sorting
        sort_field = getattr(User, sort_by, User.created_at)
        if sort_order.lower() == "asc":
            query = query.order_by(asc(sort_field))
        else:
            query = query.order_by(desc(sort_field))

        # Apply pagination
        users = query.offset((page - 1) * page_size).limit(page_size).all()
        total_pages = (total_users + page_size - 1) // page_size

        logger.info(f"Retrieved users list. Page {page}/{total_pages}, Total users: {total_users}")
        
        return UserList(
            total=total_users,
            users=users,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )

# Add statistics endpoint
class UserStats(BaseModel):
    total_users: int
    active_today: int
    created_this_week: int
    created_this_month: int

@router.get("/users/stats", response_model=UserStats)
async def get_user_stats(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get user statistics."""
    try:
        from sqlalchemy import func
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        today = now.date()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        # Total users
        total_users = db.query(func.count(User.id)).scalar()

        # Users created in last day/week/month
        created_today = db.query(func.count(User.id)).filter(
            func.date(User.created_at) == today
        ).scalar()

        created_this_week = db.query(func.count(User.id)).filter(
            User.created_at >= week_ago
        ).scalar()

        created_this_month = db.query(func.count(User.id)).filter(
            User.created_at >= month_ago
        ).scalar()

        logger.info(f"Retrieved user statistics. Total users: {total_users}")
        
        return UserStats(
            total_users=total_users,
            active_today=created_today,
            created_this_week=created_this_week,
            created_this_month=created_this_month
        )

    except Exception as e:
        logger.error(f"Error retrieving user statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user statistics: {str(e)}"
        )
