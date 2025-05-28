from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.models import Reminder
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger
import uuid

router = APIRouter()

# Pydantic models
class ReminderCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    time: datetime
    repeat_pattern: Optional[str] = None
    timezone: Optional[str] = None
    is_shared: Optional[bool] = False

class ReminderUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    time: Optional[datetime] = None
    repeat_pattern: Optional[str] = None
    timezone: Optional[str] = None
    is_shared: Optional[bool] = None

class ReminderResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    description: Optional[str]
    time: Optional[datetime]
    repeat_pattern: Optional[str]
    timezone: Optional[str]
    is_shared: Optional[bool]
    created_by: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

@router.post("/", response_model=ReminderResponse)
async def create_reminder(
    reminder_data: ReminderCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Create a new reminder."""
    try:
        user_id = current_user["uid"]
        
        reminder = Reminder(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=reminder_data.title,
            description=reminder_data.description,
            time=reminder_data.time,
            repeat_pattern=reminder_data.repeat_pattern,
            timezone=reminder_data.timezone,
            is_shared=reminder_data.is_shared,
            created_by=user_id
        )
        
        db.add(reminder)
        db.commit()
        db.refresh(reminder)
        
        logger.info(f"Created reminder: {reminder.id} for user: {user_id}")
        return ReminderResponse(
            id=str(reminder.id),
            user_id=str(reminder.user_id),
            title=reminder.title,
            description=reminder.description,
            time=reminder.time,
            repeat_pattern=reminder.repeat_pattern,
            timezone=reminder.timezone,
            is_shared=reminder.is_shared,
            created_by=str(reminder.created_by) if reminder.created_by else None,
            created_at=reminder.created_at
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Create reminder error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[ReminderResponse])
async def get_reminders(
    shared: Optional[bool] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all reminders for the current user."""
    try:
        user_id = current_user["uid"]
        query = db.query(Reminder).filter(Reminder.user_id == user_id)
        
        if shared is not None:
            query = query.filter(Reminder.is_shared == shared)
        
        reminders = query.order_by(Reminder.time).all()
        
        return [
            ReminderResponse(
                id=str(reminder.id),
                user_id=str(reminder.user_id),
                title=reminder.title,
                description=reminder.description,
                time=reminder.time,
                repeat_pattern=reminder.repeat_pattern,
                timezone=reminder.timezone,
                is_shared=reminder.is_shared,
                created_by=str(reminder.created_by) if reminder.created_by else None,
                created_at=reminder.created_at
            )
            for reminder in reminders
        ]
        
    except Exception as e:
        logger.error(f"Get reminders error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{reminder_id}", response_model=ReminderResponse)
async def get_reminder(
    reminder_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get a specific reminder."""
    try:
        user_id = current_user["uid"]
        reminder = db.query(Reminder).filter(
            Reminder.id == reminder_id,
            Reminder.user_id == user_id
        ).first()
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        return ReminderResponse(
            id=str(reminder.id),
            user_id=str(reminder.user_id),
            title=reminder.title,
            description=reminder.description,
            time=reminder.time,
            repeat_pattern=reminder.repeat_pattern,
            timezone=reminder.timezone,
            is_shared=reminder.is_shared,
            created_by=str(reminder.created_by) if reminder.created_by else None,
            created_at=reminder.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get reminder error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{reminder_id}", response_model=ReminderResponse)
async def update_reminder(
    reminder_id: str,
    reminder_data: ReminderUpdate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Update a reminder."""
    try:
        user_id = current_user["uid"]
        reminder = db.query(Reminder).filter(
            Reminder.id == reminder_id,
            Reminder.user_id == user_id
        ).first()
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Update fields if provided
        update_data = reminder_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(reminder, field, value)
        
        db.commit()
        db.refresh(reminder)
        
        logger.info(f"Updated reminder: {reminder_id}")
        return ReminderResponse(
            id=str(reminder.id),
            user_id=str(reminder.user_id),
            title=reminder.title,
            description=reminder.description,
            time=reminder.time,
            repeat_pattern=reminder.repeat_pattern,
            timezone=reminder.timezone,
            is_shared=reminder.is_shared,
            created_by=str(reminder.created_by) if reminder.created_by else None,
            created_at=reminder.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update reminder error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{reminder_id}")
async def delete_reminder(
    reminder_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Delete a reminder."""
    try:
        user_id = current_user["uid"]
        reminder = db.query(Reminder).filter(
            Reminder.id == reminder_id,
            Reminder.user_id == user_id
        ).first()
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        db.delete(reminder)
        db.commit()
        
        logger.info(f"Deleted reminder: {reminder_id}")
        return {"message": "Reminder deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete reminder error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/upcoming/today", response_model=List[ReminderResponse])
async def get_today_reminders(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get today's reminders."""
    try:
        user_id = current_user["uid"]
        today = datetime.now().date()
        
        reminders = db.query(Reminder).filter(
            Reminder.user_id == user_id,
            Reminder.time >= today,
            Reminder.time < today + timedelta(days=1)
        ).order_by(Reminder.time).all()
        
        return [
            ReminderResponse(
                id=str(reminder.id),
                user_id=str(reminder.user_id),
                title=reminder.title,
                description=reminder.description,
                time=reminder.time,
                repeat_pattern=reminder.repeat_pattern,
                timezone=reminder.timezone,
                is_shared=reminder.is_shared,
                created_by=str(reminder.created_by) if reminder.created_by else None,
                created_at=reminder.created_at
            )
            for reminder in reminders
        ]
        
    except Exception as e:
        logger.error(f"Get today's reminders error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 