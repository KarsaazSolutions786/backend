from typing import Optional
from sqlalchemy.orm import Session
from models.models import User
from connect_db import SessionLocal
from core.security import verify_password, get_password_hash
from utils.logger import logger
import uuid

class AuthService:
    """Authentication service for user management."""
    
    def __init__(self):
        pass
    
    async def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user with email and password."""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                return None
            
            # Note: This service is mainly for Firebase auth, 
            # password verification would be handled by Firebase
            return {
                "id": str(user.id),
                "email": user.email,
                "language": user.language,
                "timezone": user.timezone
            }
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
        finally:
            db.close()
    
    async def create_user(self, email: str, password: str, full_name: str = None) -> Optional[dict]:
        """Create a new user."""
        db = SessionLocal()
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(User.email == email).first()
            if existing_user:
                return None
            
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                language="en",
                timezone="UTC"
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"Created user: {email}")
            return {
                "id": str(user.id),
                "email": user.email,
                "language": user.language,
                "timezone": user.timezone,
                "is_active": True
            }
        except Exception as e:
            db.rollback()
            logger.error(f"User creation error: {e}")
            return None 
        finally:
            db.close() 