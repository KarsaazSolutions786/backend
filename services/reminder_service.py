from typing import List, Optional
from datetime import datetime
from utils.logger import logger

class ReminderService:
    """Service for managing reminders."""
    
    def __init__(self):
        self.reminders_db = {}
    
    async def create_reminder(self, user_id: str, title: str, scheduled_time: datetime) -> dict:
        """Create a new reminder."""
        reminder_id = f"reminder_{len(self.reminders_db) + 1}"
        reminder = {
            "id": reminder_id,
            "user_id": user_id,
            "title": title,
            "scheduled_time": scheduled_time,
            "created_at": datetime.utcnow()
        }
        self.reminders_db[reminder_id] = reminder
        return reminder
    
    async def get_user_reminders(self, user_id: str) -> List[dict]:
        """Get all reminders for a user."""
        return [r for r in self.reminders_db.values() if r["user_id"] == user_id] 