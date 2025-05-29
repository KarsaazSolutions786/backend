from typing import Optional
from sqlalchemy.orm import Session
from models.models import User
from connect_db import SessionLocal
from utils.logger import logger

class UserService:
    """Service for managing user operations."""
    
    def __init__(self):
        pass
    
    async def get_user_profile(self, user_id: str) -> dict:
        """Get user profile."""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                return {
                    "id": str(user.id),
                    "email": user.email,
                    "language": user.language,
                    "timezone": user.timezone,
                    "created_at": user.created_at
                }
            return {"error": "User not found"}
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {"error": str(e)}
        finally:
            db.close()
    
    async def update_user_profile(self, user_id: str, update_data: dict) -> dict:
        """Update user profile."""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            # Update fields if provided
            for field, value in update_data.items():
                if hasattr(user, field):
                    setattr(user, field, value)
            
            db.commit()
            db.refresh(user)
            
            logger.info(f"Updated user profile for {user_id}")
            return {
                "id": str(user.id),
                "email": user.email,
                "language": user.language,
                "timezone": user.timezone,
                "updated": True
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user profile: {e}")
            return {"error": str(e)}
        finally:
            db.close() 