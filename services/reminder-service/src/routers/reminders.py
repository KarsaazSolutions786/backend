from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from ..database import get_db
from ..models import Reminder, ReminderNotification, ReminderShare
from ..services.reminder_service import ReminderService
from ..services.auth_service import AuthService
from ..services.event_publisher import EventPublisher

logger = logging.getLogger(__name__)
router = APIRouter()

# Security scheme for JWT Bearer authentication
security = HTTPBearer()

# Pydantic models
class ReminderCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    time: datetime
    repeat_pattern: str = Field(default="none", pattern="^(none|daily|weekly|monthly|yearly)$")
    timezone: str = Field(default="UTC")
    priority: str = Field(default="medium", pattern="^(low|medium|high|urgent)$")
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None

class ReminderUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    time: Optional[datetime] = None
    repeat_pattern: Optional[str] = Field(None, pattern="^(none|daily|weekly|monthly|yearly)$")
    timezone: Optional[str] = None
    priority: Optional[str] = Field(None, pattern="^(low|medium|high|urgent)$")
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None

class ReminderResponse(BaseModel):
    id: str
    user_id: str
    title: str
    description: Optional[str]
    time: datetime
    repeat_pattern: str
    timezone: str
    priority: str
    category: Optional[str]
    tags: Optional[List[str]]
    is_completed: bool
    completed_at: Optional[datetime]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    next_occurrence: Optional[datetime]
    occurrence_count: str
    max_occurrences: Optional[str]

class ReminderShareCreate(BaseModel):
    shared_with_user_id: str
    can_edit: bool = False
    can_complete: bool = True
    can_reschedule: bool = False

class ReminderShareResponse(BaseModel):
    id: str
    reminder_id: str
    owner_user_id: str
    shared_with_user_id: str
    can_edit: bool
    can_complete: bool
    can_reschedule: bool
    status: str
    shared_at: datetime
    responded_at: Optional[datetime]

# Initialize services
reminder_service = ReminderService()
auth_service = AuthService()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Validate JWT (or stub) and return user info"""

    token = credentials.credentials
    user_info = await auth_service.validate_token(token)
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return user_info

def reminder_to_response(reminder) -> ReminderResponse:
    """Convert a Reminder model to ReminderResponse"""
    # Convert priority_id to priority string
    priority_str = "medium"  # default
    if reminder.priority and reminder.priority.label:
        priority_mapping = {
            "Low": "low",
            "Medium": "medium", 
            "High": "high"
        }
        priority_str = priority_mapping.get(reminder.priority.label, "medium")
    
    # Convert timezone_id to timezone string
    timezone_str = "UTC"  # default
    if reminder.timezone and reminder.timezone.name:
        timezone_str = reminder.timezone.name
    
    # Convert repeat_pattern_id to repeat_pattern string
    repeat_pattern_str = "none"  # default
    if reminder.repeat_pattern_id is not None:
        repeat_mapping = {
            0: "none",
            1: "daily",
            2: "weekly", 
            3: "monthly",
            4: "yearly"
        }
        repeat_pattern_str = repeat_mapping.get(reminder.repeat_pattern_id, "none")
    
    return ReminderResponse(
        id=str(reminder.id),
        user_id=str(reminder.customer_id),  # Convert customer_id back to user_id for API
        title=reminder.title or "",
        description=reminder.description,
        time=reminder.time,
        repeat_pattern=repeat_pattern_str,
        timezone=timezone_str,
        priority=priority_str,
        category="",  # Not implemented yet
        tags=[],  # Not implemented yet
        is_completed=reminder.is_completed,
        completed_at=reminder.completed_at,
        is_active=reminder.is_active,
        created_at=reminder.created_at,
        updated_at=reminder.updated_at,
        next_occurrence=reminder.next_occurrence,
        occurrence_count=str(reminder.occurrence_count),
        max_occurrences=str(reminder.max_occurrence) if reminder.max_occurrence else None
    )

@router.post("/", response_model=ReminderResponse, status_code=status.HTTP_201_CREATED)
async def create_reminder(
    reminder_data: ReminderCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new reminder"""
    try:
        reminder = await reminder_service.create_reminder(
            db=db,
            user_id=current_user["user_id"],
            reminder_data=reminder_data.dict()
        )
        
        # Publish event (stub for now)
        # event_publisher = request.app.state.event_publisher
        # await event_publisher.publish_event(...)
        
        logger.info(f"Reminder created: {reminder.id} for user: {current_user['user_id']}")
        
        return reminder_to_response(reminder)
    
    except Exception as e:
        logger.error(f"Error creating reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create reminder"
        )

@router.get("/", response_model=List[ReminderResponse])
async def get_reminders(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    status: Optional[str] = Query("all", regex="^(all|active|completed|pending)$"),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None)
):
    """Get user's reminders with filtering options"""
    try:
        filters = {
            "category": category,
            "priority": priority,
            "status": status,
            "from_date": from_date,
            "to_date": to_date
        }
        
        reminders = await reminder_service.get_user_reminders(
            db=db,
            user_id=current_user["user_id"],
            skip=skip,
            limit=limit,
            filters=filters
        )
        
        return [reminder_to_response(reminder) for reminder in reminders]
    
    except Exception as e:
        logger.error(f"Error getting reminders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get reminders"
        )

@router.get("/{reminder_id}", response_model=ReminderResponse)
async def get_reminder(
    reminder_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific reminder"""
    try:
        reminder = await reminder_service.get_reminder(
            db=db,
            reminder_id=reminder_id,
            user_id=current_user["user_id"]
        )
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        return reminder_to_response(reminder)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get reminder"
        )

@router.put("/{reminder_id}", response_model=ReminderResponse)
async def update_reminder(
    reminder_id: str,
    reminder_data: ReminderUpdate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a reminder"""
    try:
        # Get existing reminder
        existing_reminder = await reminder_service.get_reminder(
            db=db,
            reminder_id=reminder_id,
            user_id=current_user["user_id"]
        )
        
        if not existing_reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Update reminder
        updated_reminder = await reminder_service.update_reminder(
            db=db,
            reminder_id=reminder_id,
            user_id=current_user["user_id"],
            update_data=reminder_data.dict(exclude_unset=True)
        )
        
        # Publish event (commented out for now)
        # event_publisher = request.app.state.event_publisher
        # await event_publisher.publish_event(...)
        
        logger.info(f"Reminder updated: {reminder_id} by user: {current_user['user_id']}")
        
        return reminder_to_response(updated_reminder)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update reminder"
        )

@router.delete("/{reminder_id}")
async def delete_reminder(
    reminder_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a reminder"""
    try:
        success = await reminder_service.delete_reminder(
            db=db,
            reminder_id=reminder_id,
            user_id=current_user["user_id"]
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Publish event (commented out for now)
        # event_publisher = request.app.state.event_publisher
        # await event_publisher.publish_event(...)
        
        logger.info(f"Reminder deleted: {reminder_id} by user: {current_user['user_id']}")
        
        return {"message": "Reminder deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete reminder"
        )

@router.post("/{reminder_id}/complete")
async def complete_reminder(
    reminder_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Mark a reminder as completed"""
    try:
        reminder = await reminder_service.complete_reminder(
            db=db,
            reminder_id=reminder_id,
            user_id=current_user["user_id"]
        )
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Publish event (commented out for now)
        # event_publisher = request.app.state.event_publisher
        # await event_publisher.publish_event(...)
        
        logger.info(f"Reminder completed: {reminder_id} by user: {current_user['user_id']}")
        
        return {"message": "Reminder marked as completed"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete reminder"
        )

@router.post("/{reminder_id}/snooze")
async def snooze_reminder(
    reminder_id: str,
    request: Request,
    snooze_minutes: int = Query(..., ge=1, le=10080),  # Max 1 week
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Snooze a reminder for specified minutes"""
    try:
        reminder = await reminder_service.snooze_reminder(
            db=db,
            reminder_id=reminder_id,
            user_id=current_user["user_id"],
            snooze_minutes=snooze_minutes
        )
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Publish event (commented out for now)
        # event_publisher = request.app.state.event_publisher
        # await event_publisher.publish_event(...)
        
        logger.info(f"Reminder snoozed: {reminder_id} for {snooze_minutes} minutes by user: {current_user['user_id']}")
        
        return {
            "message": "Reminder snoozed successfully",
            "new_time": reminder.time.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error snoozing reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to snooze reminder"
        )

@router.post("/{reminder_id}/share", response_model=ReminderShareResponse)
async def share_reminder(
    reminder_id: str,
    share_data: ReminderShareCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Share a reminder with another user"""
    try:
        # Verify reminder exists and belongs to user
        reminder = await reminder_service.get_reminder(
            db=db,
            reminder_id=reminder_id,
            user_id=current_user["user_id"]
        )
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Create share
        share = await reminder_service.share_reminder(
            db=db,
            reminder_id=reminder_id,
            owner_user_id=current_user["user_id"],
            shared_with_user_id=share_data.shared_with_user_id,
            permissions={
                "can_edit": share_data.can_edit,
                "can_complete": share_data.can_complete,
                "can_reschedule": share_data.can_reschedule
            }
        )
        
        # Publish event (commented out for now)
        # event_publisher = request.app.state.event_publisher
        # await event_publisher.publish_event(...)
        
        logger.info(f"Reminder shared: {reminder_id} with user: {share_data.shared_with_user_id}")
        
        return ReminderShareResponse(
            id=str(share.id),
            reminder_id=str(share.reminder_id),
            owner_user_id=share.owner_user_id,
            shared_with_user_id=share.shared_with_user_id,
            can_edit=share.can_edit,
            can_complete=share.can_complete,
            can_reschedule=share.can_reschedule,
            status=share.status,
            shared_at=share.shared_at,
            responded_at=share.responded_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to share reminder"
        )

@router.get("/shared/with-me", response_model=List[ReminderResponse])
async def get_shared_reminders(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get reminders shared with the current user"""
    try:
        reminders = await reminder_service.get_shared_with_user(
            db=db,
            user_id=current_user["user_id"],
            skip=skip,
            limit=limit
        )
        
        return [reminder_to_response(reminder) for reminder in reminders]
    
    except Exception as e:
        logger.error(f"Error getting shared reminders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get shared reminders"
        )

@router.get("/due/upcoming")
async def get_upcoming_reminders(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    hours: int = Query(24, ge=1, le=168)  # Max 1 week
):
    """Get reminders due in the next X hours"""
    try:
        from_time = datetime.utcnow()
        to_time = from_time + timedelta(hours=hours)
        
        reminders = await reminder_service.get_reminders_in_timeframe(
            db=db,
            user_id=current_user["user_id"],
            from_time=from_time,
            to_time=to_time
        )
        
        return [
            {
                "id": str(reminder.id),
                "title": reminder.title,
                "time": reminder.time,
                "priority": reminder.priority,
                "category": reminder.category
            }
            for reminder in reminders
        ]
    
    except Exception as e:
        logger.error(f"Error getting upcoming reminders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get upcoming reminders"
        ) 