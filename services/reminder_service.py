from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.models import Reminder
from connect_db import SessionLocal
from utils.logger import logger
import uuid

class ReminderService:
    """Service for managing reminders."""
    
    def __init__(self):
        pass
    
    async def create_reminder(self, user_id: str, title: str, scheduled_time: datetime) -> dict:
        """Create a new reminder."""
        db = SessionLocal()
        try:
            reminder = Reminder(
                id=str(uuid.uuid4()),
                user_id=user_id,
                title=title,
                time=scheduled_time,
                created_by=user_id
            )
            
            db.add(reminder)
            db.commit()
            db.refresh(reminder)
            
            logger.info(f"Created reminder {reminder.id} for user {user_id}")
            return {
                "id": str(reminder.id),
                "user_id": str(reminder.user_id),
                "title": reminder.title,
                "scheduled_time": reminder.time,
                "created_at": reminder.created_at
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating reminder: {e}")
            raise
        finally:
            db.close()
    
    async def get_user_reminders(self, user_id: str) -> List[dict]:
        """Get all reminders for a user."""
        db = SessionLocal()
        try:
            reminders = db.query(Reminder).filter(Reminder.user_id == user_id).all()
            return [
                {
                    "id": str(r.id),
                    "user_id": str(r.user_id),
                    "title": r.title,
                    "scheduled_time": r.time,
                    "created_at": r.created_at
                }
                for r in reminders
            ]
        except Exception as e:
            logger.error(f"Error getting reminders: {e}")
            return []
        finally:
            db.close() 