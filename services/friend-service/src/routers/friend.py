# Use secure shared authentication 
import sys
import os

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
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
import logging
import uuid
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Customer, Friendship, FriendRequestHistory

router = APIRouter()
logger = logging.getLogger(__name__)

class FriendRequest(BaseModel):
    friend_email: str = Field(..., min_length=1, max_length=255)
    message: Optional[str] = Field(None, max_length=500)
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

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

class FriendRequestHistoryResponse(BaseModel):
    id: int
    requester_id: int
    requested_id: int
    action: str
    message: Optional[str]
    created_at: datetime
    requester_email: Optional[str]
    requested_email: Optional[str]

# Helper functions
def get_customer_by_email(db: Session, email: str) -> Optional[Customer]:
    """Get customer by email from database"""
    return db.query(Customer).filter(Customer.email == email, Customer.is_active == True).first()

def get_customer_by_id(db: Session, customer_id: int) -> Optional[Customer]:
    """Get customer by ID from database"""
    return db.query(Customer).filter(Customer.id == customer_id, Customer.is_active == True).first()

def ensure_customer_exists(db: Session, customer_id: int, email: str = None) -> Customer:
    """Ensure customer exists in database, create if not found"""
    customer = get_customer_by_id(db, customer_id)
    if not customer:
        # Create minimal customer record
        customer = Customer(
            id=customer_id,
            email=email or f"user{customer_id}@eindr.com",
            is_active=True
        )
        db.add(customer)
        db.commit()
        db.refresh(customer)
        logger.info(f"Created customer record for customer_id: {customer_id}")
    return customer

@router.post("/requests", response_model=FriendResponse)
async def send_friend_request(
    request_data: FriendRequest,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Send a friend request"""
    try:
        # Validate email format
        if not FriendRequest.validate_email(request_data.friend_email):
            raise HTTPException(
                status_code=400, 
                detail="Invalid email format. Please provide a valid email address."
            )
        
        # Ensure current user exists in database
        current_customer = ensure_customer_exists(db, customer_id)
        
        # Find friend by email
        friend_customer = get_customer_by_email(db, request_data.friend_email)
        if not friend_customer:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if trying to add self
        if friend_customer.id == customer_id:
            raise HTTPException(status_code=400, detail="Cannot add yourself as friend")
        
        # Check if friendship already exists
        existing_friendship = db.query(Friendship).filter(
            ((Friendship.customer_id == customer_id) & (Friendship.friend_id == friend_customer.id)) |
            ((Friendship.customer_id == friend_customer.id) & (Friendship.friend_id == customer_id))
        ).first()
        
        if existing_friendship:
            if existing_friendship.status == "pending":
                raise HTTPException(status_code=400, detail="Friend request already pending")
            elif existing_friendship.status == "accepted":
                raise HTTPException(status_code=400, detail="Already friends")
            elif existing_friendship.status == "blocked":
                raise HTTPException(status_code=400, detail="Cannot send friend request")
        
        # Create new friendship
        new_friendship = Friendship(
            customer_id=customer_id,
            friend_id=friend_customer.id,
            status="pending",
            initiated_by="customer",
            message=request_data.message,
            created_at=datetime.utcnow()
        )
        
        db.add(new_friendship)
        
        # Create history record for sending friend request
        history_record = FriendRequestHistory(
            requester_id=customer_id,
            requested_id=friend_customer.id,
            action="sent",
            message=request_data.message,
            created_at=datetime.utcnow()
        )
        
        db.add(history_record)
        db.commit()
        db.refresh(new_friendship)
        
        # Return response
        return FriendResponse(
            id=str(new_friendship.id),
            customer_id=str(new_friendship.customer_id),
            friend_id=str(new_friendship.friend_id),
            friend_name=friend_customer.email.split('@')[0],  # Use email prefix as name for now
            friend_email=friend_customer.email,
            status=new_friendship.status,
            created_at=new_friendship.created_at,
            accepted_at=new_friendship.accepted_at
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error in friend request: {e}")
        raise HTTPException(status_code=400, detail="Invalid request data")
    except Exception as e:
        logger.error(f"Error sending friend request: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to send friend request")

@router.get("/", response_model=List[FriendResponse])
async def get_friends(
    status: Optional[str] = Query(None),
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get customer's friends"""
    try:
        # Get friendships where user is either customer or friend
        query = db.query(Friendship).filter(
            (Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id)
        )
        
        if status:
            query = query.filter(Friendship.status == status)
        
        friendships = query.all()
        
        result = []
        for friendship in friendships:
            # Determine the friend's details
            if friendship.customer_id == customer_id:
                friend_id = friendship.friend_id
            else:
                friend_id = friendship.customer_id
            
            friend_customer = get_customer_by_id(db, friend_id)
            if friend_customer:
                result.append(FriendResponse(
                    id=str(friendship.id),
                    customer_id=str(customer_id),
                    friend_id=str(friend_id),
                    friend_name=friend_customer.email.split('@')[0],
                    friend_email=friend_customer.email,
                    status=friendship.status,
                    created_at=friendship.created_at,
                    accepted_at=friendship.accepted_at
                ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting friends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get friends")

@router.put("/requests/{friendship_id}/accept")
async def accept_friend_request(
    friendship_id: str,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Accept a friend request"""
    try:
        # Convert friendship_id to int
        try:
            friendship_id_int = int(friendship_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid friendship ID")
        
        # Find the friendship
        friendship = db.query(Friendship).filter(Friendship.id == friendship_id_int).first()
        if not friendship:
            raise HTTPException(status_code=404, detail="Friend request not found")
        
        # Check if the current user is the recipient of the friend request
        if friendship.friend_id != customer_id:
            raise HTTPException(status_code=403, detail="You can only accept friend requests sent to you")
        
        # Check if the request is still pending
        if friendship.status != "pending":
            raise HTTPException(status_code=400, detail="Friend request is not pending")
        
        # Accept the friend request
        friendship.status = "accepted"
        friendship.accepted_at = datetime.utcnow()
        
        # Create history record for accepting friend request
        history_record = FriendRequestHistory(
            requester_id=friendship.customer_id,
            requested_id=customer_id,
            action="accepted",
            created_at=datetime.utcnow()
        )
        
        db.add(history_record)
        db.commit()
        
        return {"message": "Friend request accepted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accepting friend request: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to accept friend request")

@router.delete("/requests/{friendship_id}")
async def decline_friend_request(
    friendship_id: str,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Decline or remove friend"""
    try:
        # Convert friendship_id to int
        try:
            friendship_id_int = int(friendship_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid friendship ID")
        
        # Find the friendship
        friendship = db.query(Friendship).filter(Friendship.id == friendship_id_int).first()
        if not friendship:
            raise HTTPException(status_code=404, detail="Friend request not found")
        
        # Check if the current user is involved in this friendship
        if friendship.customer_id != customer_id and friendship.friend_id != customer_id:
            raise HTTPException(status_code=403, detail="You can only manage your own friend requests")
        
        # Create history record for declining friend request
        action = "declined" if friendship.friend_id == customer_id else "canceled"
        history_record = FriendRequestHistory(
            requester_id=friendship.customer_id,
            requested_id=friendship.friend_id,
            action=action,
            created_at=datetime.utcnow()
        )
        
        db.add(history_record)
        
        # Delete the friendship
        db.delete(friendship)
        db.commit()
        
        return {"message": "Friend request declined"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error declining friend request: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to decline friend request")

@router.get("/stats", response_model=FriendshipStats)
async def get_friendship_stats(
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get friendship statistics"""
    try:
        # Get all friendships involving the user
        user_friendships = db.query(Friendship).filter(
            (Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id)
        ).all()
        
        # Calculate statistics
        total_friends = len([f for f in user_friendships if f.status == "accepted"])
        pending_requests = len([f for f in user_friendships if f.status == "pending" and f.friend_id == customer_id])
        sent_requests = len([f for f in user_friendships if f.status == "pending" and f.customer_id == customer_id])
        
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

@router.get("/history", response_model=List[FriendRequestHistoryResponse])
async def get_friend_request_history(
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get friend request history for the current user"""
    try:
        # Get all history records where user is either requester or requested
        history_records = db.query(FriendRequestHistory).filter(
            (FriendRequestHistory.requester_id == customer_id) | 
            (FriendRequestHistory.requested_id == customer_id)
        ).order_by(FriendRequestHistory.created_at.desc()).all()
        
        result = []
        for record in history_records:
            # Get requester and requested user details
            requester = get_customer_by_id(db, record.requester_id)
            requested = get_customer_by_id(db, record.requested_id)
            
            result.append(FriendRequestHistoryResponse(
                id=record.id,
                requester_id=record.requester_id,
                requested_id=record.requested_id,
                action=record.action,
                message=record.message,
                created_at=record.created_at,
                requester_email=requester.email if requester else None,
                requested_email=requested.email if requested else None
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting friend request history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get friend request history")
