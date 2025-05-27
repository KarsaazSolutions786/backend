from utils.logger import logger

class FriendService:
    """Service for managing friendships."""
    
    def __init__(self):
        self.friendships_db = {}
    
    async def add_friend(self, user_id: str, friend_email: str) -> dict:
        """Add a friend."""
        friendship_id = f"friendship_{len(self.friendships_db) + 1}"
        friendship = {
            "id": friendship_id,
            "user_id": user_id,
            "friend_email": friend_email,
            "status": "pending"
        }
        self.friendships_db[friendship_id] = friendship
        return friendship 