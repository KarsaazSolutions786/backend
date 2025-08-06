# Use secure shared authentication 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../shared'))

try:
    from simple_auth import get_current_customer_id
except ImportError:
    # Fallback to local secure implementation with audience and issuer validation
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi import Depends, HTTPException, status
    import jwt
    
    def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
        """Secure local implementation as fallback with audience and issuer validation"""
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
                audience="eindr-api",
                issuer="eindr-issuer",
                options={"verify_signature": True, "verify_exp": True, "verify_aud": True, "verify_iss": True}
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
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
import logging
import uuid
# import httpx  # Removed - using direct database queries instead of API calls
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
    received_requests: int
    blocked_friends: int
    shared_reminders: int
    shared_notes: int
    mutual_friends: int

class BlockFriendRequest(BaseModel):
    reason: Optional[str] = Field(None, max_length=500)

class UnblockFriendRequest(BaseModel):
    pass

class SearchFriendsRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)
    limit: Optional[int] = Field(10, ge=1, le=50)

class MutualFriendsResponse(BaseModel):
    mutual_friends: List[FriendResponse]
    count: int

class FriendRequestHistoryResponse(BaseModel):
    id: int
    requester_id: int
    requested_id: int
    action: str
    message: Optional[str]
    created_at: datetime
    requester_email: Optional[str]
    requested_email: Optional[str]

class SuggestedUser(BaseModel):
    id: int
    email: str
    display_name: Optional[str] = None
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    is_verified: bool = False
    created_at: datetime
    friendship_status: Optional[str] = None  # None, 'pending_sent', 'pending_received', 'friends', 'blocked'

class SuggestionsResponse(BaseModel):
    users: List[SuggestedUser]
    total: int
    page: int
    limit: int
    pages: int



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
    request: Request,
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
            created_at=datetime.utcnow(),
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown")
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
    status: Optional[str] = Query(None, description="Filter by status: pending, accepted, blocked, declined"),
    limit: Optional[int] = Query(50, ge=1, le=100, description="Maximum number of friends to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of friends to skip"),
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get friends list with optional status filter and pagination"""
    try:
        # Base query for friendships involving the current user
        query = db.query(Friendship).filter(
            (Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id)
        )
        
        # Apply status filter if provided
        if status:
            if status not in ["pending", "accepted", "blocked", "declined"]:
                raise HTTPException(status_code=400, detail="Invalid status. Must be one of: pending, accepted, blocked, declined")
            query = query.filter(Friendship.status == status)
        
        # Apply pagination
        friendships = query.order_by(Friendship.created_at.desc()).offset(offset).limit(limit).all()
        
        result = []
        for friendship in friendships:
            # Determine the friend's details
            if friendship.customer_id == customer_id:
                friend_id = friendship.friend_id
            else:
                friend_id = friendship.customer_id
            
            friend = get_customer_by_id(db, friend_id)
            if friend:
                result.append(FriendResponse(
                    id=str(friendship.id),
                    customer_id=str(customer_id),
                    friend_id=str(friend_id),
                    friend_name=friend.email.split('@')[0],  # Use email prefix as name for now
                    friend_email=friend.email,
                    status=friendship.status,
                    created_at=friendship.created_at,
                    accepted_at=friendship.accepted_at
                ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting mutual friends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get mutual friends")

@router.delete("/requests/{friendship_id}/cancel")
async def cancel_friend_request(
    friendship_id: str,
    request: Request,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Cancel a sent friend request"""
    try:
        # Find the pending friendship request sent by the current user
        friendship = db.query(Friendship).filter(
            Friendship.id == friendship_id,
            Friendship.customer_id == customer_id,  # Only the sender can cancel
            Friendship.status == "pending"
        ).first()
        
        if not friendship:
            raise HTTPException(status_code=404, detail="Pending friend request not found or you don't have permission to cancel it")
        
        # Create history record before deletion
        history_record = FriendRequestHistory(
            requester_id=customer_id,
            requested_id=friendship.friend_id,
            action="canceled",
            message="Friend request canceled by sender",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        db.add(history_record)
        
        # Delete the friendship record
        db.delete(friendship)
        db.commit()
        
        logger.info(f"User {customer_id} canceled friend request {friendship_id}")
        return {"message": "Friend request canceled successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error canceling friend request: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel friend request")

@router.get("/requests/incoming", response_model=List[FriendResponse])
async def get_incoming_friend_requests(
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get incoming friend requests (requests received by the current user)"""
    try:
        # Get pending friendships where current user is the friend (recipient)
        incoming_requests = db.query(Friendship).filter(
            Friendship.friend_id == customer_id,
            Friendship.status == "pending"
        ).order_by(Friendship.created_at.desc()).all()
        
        result = []
        for friendship in incoming_requests:
            # Get the requester's details
            requester = get_customer_by_id(db, friendship.customer_id)
            if requester:
                result.append(FriendResponse(
                    id=str(friendship.id),
                    customer_id=str(customer_id),
                    friend_id=str(friendship.customer_id),
                    friend_name=requester.email.split('@')[0],
                    friend_email=requester.email,
                    status=friendship.status,
                    created_at=friendship.created_at,
                    accepted_at=friendship.accepted_at
                ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting incoming friend requests: {e}")
        raise HTTPException(status_code=500, detail="Failed to get incoming friend requests")

@router.get("/requests/outgoing", response_model=List[FriendResponse])
async def get_outgoing_friend_requests(
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get outgoing friend requests (requests sent by the current user)"""
    try:
        # Get pending friendships where current user is the customer (sender)
        outgoing_requests = db.query(Friendship).filter(
            Friendship.customer_id == customer_id,
            Friendship.status == "pending"
        ).order_by(Friendship.created_at.desc()).all()
        
        result = []
        for friendship in outgoing_requests:
            # Get the recipient's details
            recipient = get_customer_by_id(db, friendship.friend_id)
            if recipient:
                result.append(FriendResponse(
                    id=str(friendship.id),
                    customer_id=str(customer_id),
                    friend_id=str(friendship.friend_id),
                    friend_name=recipient.email.split('@')[0],
                    friend_email=recipient.email,
                    status=friendship.status,
                    created_at=friendship.created_at,
                    accepted_at=friendship.accepted_at
                ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting outgoing friend requests: {e}")
        raise HTTPException(status_code=500, detail="Failed to get outgoing friend requests")

@router.get("/blocked", response_model=List[FriendResponse])
async def get_blocked_friends(
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get list of blocked friends"""
    try:
        # Get all blocked friendships
        blocked_friendships = db.query(Friendship).filter(
            ((Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id)),
            Friendship.status == "blocked"
        ).order_by(Friendship.updated_at.desc()).all()
        
        result = []
        for friendship in blocked_friendships:
            # Determine the friend's details
            if friendship.customer_id == customer_id:
                friend_id = friendship.friend_id
            else:
                friend_id = friendship.customer_id
            
            friend = get_customer_by_id(db, friend_id)
            if friend:
                result.append(FriendResponse(
                    id=str(friendship.id),
                    customer_id=str(customer_id),
                    friend_id=str(friend_id),
                    friend_name=friend.email.split('@')[0],
                    friend_email=friend.email,
                    status=friendship.status,
                    created_at=friendship.created_at,
                    accepted_at=friendship.accepted_at
                ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting blocked friends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get blocked friends")

@router.put("/requests/{friendship_id}/accept")
async def accept_friend_request(
    friendship_id: str,
    request: Request,
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
            created_at=datetime.utcnow(),
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown")
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
    request: Request,
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
            created_at=datetime.utcnow(),
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown")
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
        received_requests = pending_requests  # Same as pending_requests
        blocked_friends = len([f for f in user_friendships if f.status == "blocked"])
        
        # Calculate mutual friends count (simplified - count of friends who have mutual connections)
        user_friend_ids = set()
        for f in user_friendships:
            if f.status == "accepted":
                friend_id = f.friend_id if f.customer_id == customer_id else f.customer_id
                user_friend_ids.add(friend_id)
        
        mutual_friends_count = 0
        for friend_id in user_friend_ids:
            friend_connections = db.query(Friendship).filter(
                ((Friendship.customer_id == friend_id) | (Friendship.friend_id == friend_id)),
                Friendship.status == "accepted"
            ).count()
            if friend_connections > 1:  # More than just the connection to current user
                mutual_friends_count += 1
        
        return FriendshipStats(
            total_friends=total_friends,
            pending_requests=pending_requests,
            sent_requests=sent_requests,
            received_requests=received_requests,
            blocked_friends=blocked_friends,
            shared_reminders=0,  # Would calculate from reminder service
            shared_notes=0,      # Would calculate from note service
            mutual_friends=mutual_friends_count
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
            
            # Determine the action from the current user's perspective
            action = record.action
            if record.action == "sent" and record.requested_id == customer_id:
                # If someone sent a request to the current user, show it as "incoming"
                action = "incoming"
            
            result.append(FriendRequestHistoryResponse(
                id=record.id,
                requester_id=record.requester_id,
                requested_id=record.requested_id,
                action=action,
                message=record.message,
                created_at=record.created_at,
                requester_email=requester.email if requester else None,
                requested_email=requested.email if requested else None
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting friend request history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get friend request history")

@router.put("/requests/{friendship_id}/block")
async def block_friend(
    friendship_id: str,
    block_data: BlockFriendRequest,
    request: Request,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Block a friend or friend request"""
    try:
        # Find the friendship
        friendship = db.query(Friendship).filter(
            Friendship.id == friendship_id,
            ((Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id))
        ).first()
        
        if not friendship:
            raise HTTPException(status_code=404, detail="Friendship not found")
        
        # Update friendship status to blocked
        friendship.status = "blocked"
        friendship.updated_at = datetime.utcnow()
        
        # Create history record
        history_record = FriendRequestHistory(
            requester_id=customer_id,
            requested_id=friendship.friend_id if friendship.customer_id == customer_id else friendship.customer_id,
            action="blocked",
            message=block_data.reason,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        db.add(history_record)
        
        db.commit()
        
        logger.info(f"User {customer_id} blocked friendship {friendship_id}")
        return {"message": "Friend blocked successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error blocking friend: {e}")
        raise HTTPException(status_code=500, detail="Failed to block friend")

@router.put("/requests/{friendship_id}/unblock")
async def unblock_friend(
    friendship_id: str,
    request: Request,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Unblock a friend"""
    try:
        # Find the blocked friendship
        friendship = db.query(Friendship).filter(
            Friendship.id == friendship_id,
            ((Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id)),
            Friendship.status == "blocked"
        ).first()
        
        if not friendship:
            raise HTTPException(status_code=404, detail="Blocked friendship not found")
        
        # Update friendship status back to accepted (assuming it was accepted before blocking)
        friendship.status = "accepted"
        friendship.updated_at = datetime.utcnow()
        
        # Create history record
        history_record = FriendRequestHistory(
            requester_id=customer_id,
            requested_id=friendship.friend_id if friendship.customer_id == customer_id else friendship.customer_id,
            action="unblocked",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        db.add(history_record)
        
        db.commit()
        
        logger.info(f"User {customer_id} unblocked friendship {friendship_id}")
        return {"message": "Friend unblocked successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error unblocking friend: {e}")
        raise HTTPException(status_code=500, detail="Failed to unblock friend")

@router.post("/search", response_model=List[FriendResponse])
async def search_friends(
    search_data: SearchFriendsRequest,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Search for friends by email or name"""
    try:
        # Search for customers by email (basic search)
        customers = db.query(Customer).filter(
            Customer.email.ilike(f"%{search_data.query}%"),
            Customer.id != customer_id,  # Exclude self
            Customer.is_active == True
        ).limit(search_data.limit).all()
        
        result = []
        for customer in customers:
            # Check if there's an existing friendship
            friendship = db.query(Friendship).filter(
                ((Friendship.customer_id == customer_id) & (Friendship.friend_id == customer.id)) |
                ((Friendship.customer_id == customer.id) & (Friendship.friend_id == customer_id))
            ).first()
            
            status = "none"  # No friendship exists
            if friendship:
                status = friendship.status
            
            result.append(FriendResponse(
                id=str(friendship.id) if friendship else str(customer.id),
                customer_id=str(customer_id),
                friend_id=str(customer.id),
                friend_name=customer.email.split('@')[0],  # Use email prefix as name
                friend_email=customer.email,
                status=status,
                created_at=friendship.created_at if friendship else customer.created_at if hasattr(customer, 'created_at') else datetime.utcnow(),
                accepted_at=friendship.accepted_at if friendship else None
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error searching friends: {e}")
        raise HTTPException(status_code=500, detail="Failed to search friends")

@router.get("/mutual/{friend_id}", response_model=MutualFriendsResponse)
async def get_mutual_friends(
    friend_id: int,
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get mutual friends between current user and specified friend"""
    try:
        # Verify that the specified friend_id is actually a friend
        friendship = db.query(Friendship).filter(
            ((Friendship.customer_id == customer_id) & (Friendship.friend_id == friend_id)) |
            ((Friendship.customer_id == friend_id) & (Friendship.friend_id == customer_id)),
            Friendship.status == "accepted"
        ).first()
        
        if not friendship:
            raise HTTPException(status_code=404, detail="Friend relationship not found")
        
        # Get current user's friends
        user_friends = db.query(Friendship).filter(
            ((Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id)),
            Friendship.status == "accepted"
        ).all()
        
        # Get specified friend's friends
        friend_friends = db.query(Friendship).filter(
            ((Friendship.customer_id == friend_id) | (Friendship.friend_id == friend_id)),
            Friendship.status == "accepted"
        ).all()
        
        # Extract friend IDs
        user_friend_ids = set()
        for f in user_friends:
            if f.customer_id == customer_id:
                user_friend_ids.add(f.friend_id)
            else:
                user_friend_ids.add(f.customer_id)
        
        friend_friend_ids = set()
        for f in friend_friends:
            if f.customer_id == friend_id:
                friend_friend_ids.add(f.friend_id)
            else:
                friend_friend_ids.add(f.customer_id)
        
        # Find mutual friends (excluding self and the specified friend)
        mutual_friend_ids = user_friend_ids.intersection(friend_friend_ids)
        mutual_friend_ids.discard(customer_id)
        mutual_friend_ids.discard(friend_id)
        
        # Get customer details for mutual friends
        mutual_friends = []
        for mutual_id in mutual_friend_ids:
            customer = get_customer_by_id(db, mutual_id)
            if customer:
                # Get the friendship record for response
                friendship_record = db.query(Friendship).filter(
                    ((Friendship.customer_id == customer_id) & (Friendship.friend_id == mutual_id)) |
                    ((Friendship.customer_id == mutual_id) & (Friendship.friend_id == customer_id))
                ).first()
                
                mutual_friends.append(FriendResponse(
                    id=str(friendship_record.id),
                    customer_id=str(customer_id),
                    friend_id=str(mutual_id),
                    friend_name=customer.email.split('@')[0],
                    friend_email=customer.email,
                    status=friendship_record.status,
                    created_at=friendship_record.created_at,
                    accepted_at=friendship_record.accepted_at
                ))
        
        return MutualFriendsResponse(
            mutual_friends=mutual_friends,
            count=len(mutual_friends)
        )
        
    except Exception as e:
        logger.error(f"Error getting mutual friends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get mutual friends")

@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_suggested_users(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Number of users per page"),
    search: Optional[str] = Query(None, description="Search by name or email"),
    customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """
    Get suggested users for the suggestion tab.
    Returns all users excluding the current user and shows friendship status.
    """
    try:
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        # Build base query for customers, excluding current user
        query = db.query(Customer).filter(
            Customer.id != customer_id,
            Customer.is_active == True
        )
        
        # Apply search filter if provided
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                Customer.email.ilike(search_term)
            )
        
        # Get total count for pagination
        total = query.count()
        
        # Apply pagination and get users
        users = query.order_by(Customer.created_at.desc()).offset(offset).limit(limit).all()
        
        # Get all friendship relationships for the current user
        friendships = db.query(Friendship).filter(
            (Friendship.customer_id == customer_id) | (Friendship.friend_id == customer_id)
        ).all()
        
        # Create a mapping of user_id -> friendship_status
        friendship_status_map = {}
        for friendship in friendships:
            other_user_id = friendship.friend_id if friendship.customer_id == customer_id else friendship.customer_id
            
            if friendship.status == "accepted":
                friendship_status_map[other_user_id] = "friends"
            elif friendship.status == "pending":
                if friendship.customer_id == customer_id:
                    friendship_status_map[other_user_id] = "pending_sent"
                else:
                    friendship_status_map[other_user_id] = "pending_received"
            elif friendship.status == "blocked":
                friendship_status_map[other_user_id] = "blocked"
        
        # Process the users and add friendship status
        suggested_users = []
        for user in users:
            # Extract display name from email (before @ symbol) as fallback
            display_name = user.email.split('@')[0] if user.email else None
            
            suggested_user = SuggestedUser(
                id=user.id,
                email=user.email,
                display_name=display_name,
                full_name=None,  # Not available in current Customer model
                bio=None,  # Not available in current Customer model
                avatar_url=None,  # Not available in current Customer model
                is_verified=False,  # Not available in current Customer model
                created_at=user.created_at,
                friendship_status=friendship_status_map.get(user.id)
            )
            suggested_users.append(suggested_user)
        
        # Calculate pagination info
        pages = (total + limit - 1) // limit if total > 0 else 1
        
        return SuggestionsResponse(
            users=suggested_users,
            total=total,
            page=page,
            limit=limit,
            pages=pages
        )
        
    except Exception as e:
        logger.error(f"Error getting suggested users: {e}")
        raise HTTPException(status_code=500, detail="Failed to get suggested users")
