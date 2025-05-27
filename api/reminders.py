from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.models import Reminder
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger

router = APIRouter()

# Pydantic models
class ReminderCreate(BaseModel):
    title: str
    description: Optional[str] = None
    scheduled_time: datetime
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None  # daily, weekly, monthly

class ReminderUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    is_recurring: Optional[bool] = None
    recurrence_pattern: Optional[str] = None
    is_completed: Optional[bool] = None

class ReminderResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    scheduled_time: datetime
    is_recurring: bool
    recurrence_pattern: Optional[str]
    is_completed: bool
    user_id: str
    created_at: datetime
    updated_at: datetime

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
        reminder = Reminder(
            title=reminder_data.title,
            description=reminder_data.description,
            scheduled_time=reminder_data.scheduled_time,
            is_recurring=reminder_data.is_recurring,
            recurrence_pattern=reminder_data.recurrence_pattern,
            user_id=current_user["uid"]
        )
        
        db.add(reminder)
        db.commit()
        db.refresh(reminder)
        
        logger.info(f"Created reminder: {reminder.id} for user: {current_user['uid']}")
        return reminder
        
    except Exception as e:
        db.rollback()
        logger.error(f"Create reminder error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[ReminderResponse])
async def get_reminders(
    completed: Optional[bool] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all reminders for the current user."""
    try:
        query = db.query(Reminder).filter(Reminder.user_id == current_user["uid"])
        
        if completed is not None:
            query = query.filter(Reminder.is_completed == completed)
        
        reminders = query.order_by(Reminder.scheduled_time).all()
        return reminders
        
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
        reminder = db.query(Reminder).filter(
            Reminder.id == reminder_id,
            Reminder.user_id == current_user["uid"]
        ).first()
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        return reminder
        
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
        reminder = db.query(Reminder).filter(
            Reminder.id == reminder_id,
            Reminder.user_id == current_user["uid"]
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
        return reminder
        
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
        reminder = db.query(Reminder).filter(
            Reminder.id == reminder_id,
            Reminder.user_id == current_user["uid"]
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