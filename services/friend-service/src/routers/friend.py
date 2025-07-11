# Use secure shared authentication 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../shared'))

try:
    from simple_auth import get_current_customer_id
except ImportError:
    # Fallback to local secure implementation
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi import Depends, HTTPException, status
    import jwt
    
    def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
        """Secure local implementation as fallback"""
        try:
            token = credentials.credentials
            secret_key = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
            
            # Validate production environment
            if os.getenv("ENVIRONMENT") == "production" and secret_key in [
                "your-secret-key-here", 
                "your-secret-key-here-change-in-production",
                "eindr-super-secret-key-change-in-production-123456789"
            ]:
                raise ValueError("Production environment requires a secure SECRET_KEY")
            
            payload = jwt.decode(
                token, 
                secret_key, 
                algorithms=["HS256"],
                options={"verify_signature": True, "verify_exp": True}
            )
            customer_id = payload.get("sub")
            if customer_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing subject"
                )
            return int(customer_id)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

class FriendRequest(BaseModel):
    friend_email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    message: Optional[str] = Field(None, max_length=500)

class FriendResponse(BaseModel):
    id: str
    customer_id: str
    friend_id: str
    friend_name: str
    friend_email: str
    status: str  # 'pending', 'accepted', 'blocked'
    created_at: datetime
    accepted_at: Optional[datetime]

class FriendshipStats(BaseModel):
    total_friends: int
    pending_requests: int
    sent_requests: int
    shared_reminders: int
    shared_notes: int

# Mock storage
friendships_storage = {}
users_storage = {
    'user-123': {'name': 'John Doe', 'email': 'john@example.com'},
    'user-456': {'name': 'Jane Smith', 'email': 'jane@example.com'},
}



@router.post("/requests", response_model=FriendResponse)
async def send_friend_request(
    request_data: FriendRequest,
    customer_id: int = Depends(get_current_customer_id)
):
    """Send a friend request"""
    try:
        
        # Find friend by email (mock lookup)
        friend_id = None
        for uid, customer_data in users_storage.items():
            if customer_data['email'] == request_data.friend_email:
                friend_id = uid
                break
        
        if not friend_id:
            raise HTTPException(status_code=404, detail="User not found")
        
        if friend_id == customer_id:
            raise HTTPException(status_code=400, detail="Cannot add yourself as friend")
        
        friendship_id = str(uuid.uuid4())
        friendship = {
            "id": friendship_id,
            "customer_id": customer_id,
            "friend_id": friend_id,
            "friend_name": users_storage[friend_id]['name'],
            "friend_email": users_storage[friend_id]['email'],
            "status": "pending",
            "message": request_data.message,
            "created_at": datetime.utcnow(),
            "accepted_at": None
        }
        
        friendships_storage[friendship_id] = friendship
        
        return FriendResponse(**friendship)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending friend request: {e}")
        raise HTTPException(status_code=500, detail="Failed to send friend request")

@router.get("/", response_model=List[FriendResponse])
async def get_friends(
    status: Optional[str] = Query(None),
    customer_id: int = Depends(get_current_customer_id)
):
    """Get customer's friends"""
    try:
        user_friendships = [f for f in friendships_storage.values() 
                           if f["customer_id"] == customer_id or f["friend_id"] == customer_id]
        
        if status:
            user_friendships = [f for f in user_friendships if f["status"] == status]
        
        return [FriendResponse(**f) for f in user_friendships]
        
    except Exception as e:
        logger.error(f"Error getting friends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get friends")

@router.put("/requests/{friendship_id}/accept")
async def accept_friend_request(friendship_id: str):
    """Accept a friend request"""
    try:
        friendship = friendships_storage.get(friendship_id)
        if not friendship:
            raise HTTPException(status_code=404, detail="Friend request not found")
        
        friendship["status"] = "accepted"
        friendship["accepted_at"] = datetime.utcnow()
        
        return {"message": "Friend request accepted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accepting friend request: {e}")
        raise HTTPException(status_code=500, detail="Failed to accept friend request")

@router.delete("/requests/{friendship_id}")
async def decline_friend_request(friendship_id: str):
    """Decline or remove friend"""
    try:
        if friendship_id in friendships_storage:
            del friendships_storage[friendship_id]
        
        return {"message": "Friend request declined"}
        
    except Exception as e:
        logger.error(f"Error declining friend request: {e}")
        raise HTTPException(status_code=500, detail="Failed to decline friend request")

@router.get("/stats", response_model=FriendshipStats)
async def get_friendship_stats(customer_id: int = Depends(get_current_customer_id)):
    """Get friendship statistics"""
    try:
        user_friendships = [f for f in friendships_storage.values() 
                           if f["customer_id"] == customer_id or f["friend_id"] == customer_id]
        
        total_friends = len([f for f in user_friendships if f["status"] == "accepted"])
        pending_requests = len([f for f in user_friendships if f["status"] == "pending" and f["friend_id"] == customer_id])
        sent_requests = len([f for f in user_friendships if f["status"] == "pending" and f["customer_id"] == customer_id])
        
        return FriendshipStats(
            total_friends=total_friends,
            pending_requests=pending_requests,
            sent_requests=sent_requests,
            shared_reminders=0,  # Would calculate from reminder service
            shared_notes=0       # Would calculate from note service
        )
        
    except Exception as e:
        logger.error(f"Error getting friendship stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get friendship stats")
