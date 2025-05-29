from typing import List
from sqlalchemy.orm import Session
from models.models import Friendship, User
from connect_db import SessionLocal
from utils.logger import logger
import uuid

class FriendService:
    """Service for managing friendships."""
    
    def __init__(self):
        pass
    
    async def add_friend(self, user_id: str, friend_email: str) -> dict:
        """Add a friend."""
        db = SessionLocal()
        try:
            # Find friend by email
            friend = db.query(User).filter(User.email == friend_email).first()
            if not friend:
                return {"error": "Friend not found"}
            
            # Check if friendship already exists
            existing = db.query(Friendship).filter(
                ((Friendship.user_id == user_id) & (Friendship.friend_id == friend.id)) |
                ((Friendship.user_id == friend.id) & (Friendship.friend_id == user_id))
            ).first()
            
            if existing:
                return {"error": "Friendship already exists"}
            
            friendship = Friendship(
                id=str(uuid.uuid4()),
                user_id=user_id,
                friend_id=friend.id,
                status="pending"
            )
            
            db.add(friendship)
            db.commit()
            db.refresh(friendship)
            
            logger.info(f"Created friendship {friendship.id} between {user_id} and {friend.id}")
            return {
                "id": str(friendship.id),
                "user_id": str(friendship.user_id),
                "friend_id": str(friendship.friend_id),
            "friend_email": friend_email,
                "status": friendship.status,
                "created_at": friendship.created_at
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating friendship: {e}")
            return {"error": str(e)}
        finally:
            db.close()
    
    async def get_user_friends(self, user_id: str) -> List[dict]:
        """Get all friends for a user."""
        db = SessionLocal()
        try:
            friendships = db.query(Friendship).filter(
                (Friendship.user_id == user_id) | (Friendship.friend_id == user_id)
            ).all()
            
            result = []
            for friendship in friendships:
                friend_id = friendship.friend_id if friendship.user_id == user_id else friendship.user_id
                friend = db.query(User).filter(User.id == friend_id).first()
                
                if friend:
                    result.append({
                        "id": str(friendship.id),
                        "friend_id": str(friend.id),
                        "friend_email": friend.email,
                        "status": friendship.status,
                        "created_at": friendship.created_at
                    })
            
            return result
        except Exception as e:
            logger.error(f"Error getting friendships: {e}")
            return []
        finally:
            db.close() 