from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from models.models import Friendship, User
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger
import datetime

router = APIRouter()

class FriendRequest(BaseModel):
    friend_email: str

class FriendResponse(BaseModel):
    id: str
    user_id: str
    friend_id: str
    status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        from_attributes = True

@router.post("/add", response_model=FriendResponse)
async def add_friend(
    friend_data: FriendRequest,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Send a friend request."""
    try:
        # Find friend by email
        friend = db.query(User).filter(User.email == friend_data.friend_email).first()
        if not friend:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found with this email"
            )
        
        # Check if friendship already exists
        existing_friendship = db.query(Friendship).filter(
            ((Friendship.user_id == current_user["uid"]) & (Friendship.friend_id == friend.id)) |
            ((Friendship.user_id == friend.id) & (Friendship.friend_id == current_user["uid"]))
        ).first()
        
        if existing_friendship:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Friendship already exists"
            )
        
        # Create new friendship
        friendship = Friendship(
            user_id=current_user["uid"],
            friend_id=friend.id,
            status="pending"
        )
        
        db.add(friendship)
        db.commit()
        db.refresh(friendship)
        
        logger.info(f"Friend request sent: {current_user['uid']} -> {friend.id}")
        return friendship
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Add friend error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[FriendResponse])
async def get_friends(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get user's friends list."""
    try:
        friendships = db.query(Friendship).filter(
            (Friendship.user_id == current_user["uid"]) |
            (Friendship.friend_id == current_user["uid"])
        ).all()
        
        return friendships
        
    except Exception as e:
        logger.error(f"Get friends error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{friendship_id}/accept")
async def accept_friend_request(
    friendship_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Accept a friend request."""
    try:
        friendship = db.query(Friendship).filter(
            Friendship.id == friendship_id,
            Friendship.friend_id == current_user["uid"],
            Friendship.status == "pending"
        ).first()
        
        if not friendship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Friend request not found"
            )
        
        friendship.status = "accepted"
        db.commit()
        
        logger.info(f"Friend request accepted: {friendship_id}")
        return {"message": "Friend request accepted"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Accept friend request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{friendship_id}/reject")
async def reject_friend_request(
    friendship_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Reject a friend request."""
    try:
        friendship = db.query(Friendship).filter(
            Friendship.id == friendship_id,
            Friendship.friend_id == current_user["uid"],
            Friendship.status == "pending"
        ).first()
        
        if not friendship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Friend request not found"
            )
        
        friendship.status = "rejected"
        db.commit()
        
        logger.info(f"Friend request rejected: {friendship_id}")
        return {"message": "Friend request rejected"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Reject friend request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 