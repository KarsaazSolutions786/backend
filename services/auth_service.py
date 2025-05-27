from typing import Optional
from core.security import verify_password, get_password_hash
from utils.logger import logger

class AuthService:
    """Authentication service for user management."""
    
    def __init__(self):
        # In production, this would connect to a real database
        self.users_db = {}
    
    async def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user with email and password."""
        try:
            user = self.users_db.get(email)
            if not user:
                return None
            
            if not verify_password(password, user["hashed_password"]):
                return None
            
            return user
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def create_user(self, email: str, password: str, full_name: str = None) -> Optional[dict]:
        """Create a new user."""
        try:
            if email in self.users_db:
                return None
            
            user = {
                "id": str(len(self.users_db) + 1),
                "email": email,
                "full_name": full_name,
                "hashed_password": get_password_hash(password),
                "is_active": True
            }
            
            self.users_db[email] = user
            logger.info(f"Created user: {email}")
            return user
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return None 