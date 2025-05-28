from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.models import Friendship, User, Permission
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger
import uuid

router = APIRouter()

# Pydantic models
class FriendshipCreate(BaseModel):
    friend_id: str
    status: Optional[str] = "pending"

class FriendshipUpdate(BaseModel):
    status: Optional[str] = None

class PermissionCreate(BaseModel):
    friend_id: str
    auto_accept_reminders: Optional[bool] = False
    auto_accept_notes: Optional[bool] = False

class PermissionUpdate(BaseModel):
    auto_accept_reminders: Optional[bool] = None
    auto_accept_notes: Optional[bool] = None

class UserBasicResponse(BaseModel):
    id: str
    email: str
    language: Optional[str]
    timezone: Optional[str]

    class Config:
        from_attributes = True

class FriendshipResponse(BaseModel):
    id: str
    user_id: str
    friend_id: str
    status: str
    created_at: datetime
    friend_info: Optional[UserBasicResponse] = None

    class Config:
        from_attributes = True

class PermissionResponse(BaseModel):
    id: str
    user_id: str
    friend_id: str
    auto_accept_reminders: Optional[bool]
    auto_accept_notes: Optional[bool]
    updated_at: datetime
    friend_info: Optional[UserBasicResponse] = None

    class Config:
        from_attributes = True

@router.post("/", response_model=FriendshipResponse)
async def send_friend_request(
    friendship_data: FriendshipCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Send a friend request."""
    try:
        # Validate status
        if friendship_data.status not in ['pending', 'accepted', 'blocked']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status must be 'pending', 'accepted', or 'blocked'"
            )
        
        user_id = current_user["uid"]
        friend_id = friendship_data.friend_id
        
        # Check if friend exists
        friend = db.query(User).filter(User.id == friend_id).first()
        if not friend:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Friend user not found"
            )
        
        # Check if friendship already exists
        existing_friendship = db.query(Friendship).filter(
            ((Friendship.user_id == user_id) & (Friendship.friend_id == friend_id)) |
            ((Friendship.user_id == friend_id) & (Friendship.friend_id == user_id))
        ).first()
        
        if existing_friendship:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Friendship already exists"
            )
        
        friendship = Friendship(
            id=str(uuid.uuid4()),
            user_id=user_id,
            friend_id=friend_id,
            status=friendship_data.status
        )
        
        db.add(friendship)
        db.commit()
        db.refresh(friendship)
        
        logger.info(f"Created friendship: {friendship.id} between {user_id} and {friend_id}")
        
        return FriendshipResponse(
            id=str(friendship.id),
            user_id=str(friendship.user_id),
            friend_id=str(friendship.friend_id),
            status=friendship.status,
            created_at=friendship.created_at,
            friend_info=UserBasicResponse(
                id=str(friend.id),
                email=friend.email,
                language=friend.language,
                timezone=friend.timezone
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Create friendship error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[FriendshipResponse])
async def get_friendships(
    status_filter: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all friendships for the current user."""
    try:
        user_id = current_user["uid"]
        
        # Get friendships where user is either the requester or the friend
        query = db.query(Friendship).filter(
            (Friendship.user_id == user_id) | (Friendship.friend_id == user_id)
        )
        
        if status_filter:
            if status_filter not in ['pending', 'accepted', 'blocked']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Status must be 'pending', 'accepted', or 'blocked'"
                )
            query = query.filter(Friendship.status == status_filter)
        
        friendships = query.order_by(Friendship.created_at.desc()).all()
        
        result = []
        for friendship in friendships:
            # Determine the friend's ID (the other user in the relationship)
            friend_id = friendship.friend_id if friendship.user_id == user_id else friendship.user_id
            friend = db.query(User).filter(User.id == friend_id).first()
            
            friend_info = None
            if friend:
                friend_info = UserBasicResponse(
                    id=str(friend.id),
                    email=friend.email,
                    language=friend.language,
                    timezone=friend.timezone
                )
            
            result.append(FriendshipResponse(
                id=str(friendship.id),
                user_id=str(friendship.user_id),
                friend_id=str(friendship.friend_id),
                status=friendship.status,
                created_at=friendship.created_at,
                friend_info=friend_info
            ))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get friendships error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{friendship_id}", response_model=FriendshipResponse)
async def update_friendship(
    friendship_id: str,
    friendship_data: FriendshipUpdate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Update a friendship (accept, reject, block)."""
    try:
        if friendship_data.status and friendship_data.status not in ['pending', 'accepted', 'blocked']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status must be 'pending', 'accepted', or 'blocked'"
            )
        
        user_id = current_user["uid"]
        
        # Find friendship where user is involved
        friendship = db.query(Friendship).filter(
            Friendship.id == friendship_id,
            (Friendship.user_id == user_id) | (Friendship.friend_id == user_id)
        ).first()
        
        if not friendship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Friendship not found"
            )
        
        # Update fields if provided
        update_data = friendship_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(friendship, field, value)
        
        db.commit()
        db.refresh(friendship)
        
        # Get friend info
        friend_id = friendship.friend_id if friendship.user_id == user_id else friendship.user_id
        friend = db.query(User).filter(User.id == friend_id).first()
        
        friend_info = None
        if friend:
            friend_info = UserBasicResponse(
                id=str(friend.id),
                email=friend.email,
                language=friend.language,
                timezone=friend.timezone
            )
        
        logger.info(f"Updated friendship: {friendship_id}")
        return FriendshipResponse(
            id=str(friendship.id),
            user_id=str(friendship.user_id),
            friend_id=str(friendship.friend_id),
            status=friendship.status,
            created_at=friendship.created_at,
            friend_info=friend_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update friendship error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{friendship_id}")
async def delete_friendship(
    friendship_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Delete a friendship."""
    try:
        user_id = current_user["uid"]
        
        friendship = db.query(Friendship).filter(
            Friendship.id == friendship_id,
            (Friendship.user_id == user_id) | (Friendship.friend_id == user_id)
        ).first()
        
        if not friendship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Friendship not found"
            )
        
        db.delete(friendship)
        db.commit()
        
        logger.info(f"Deleted friendship: {friendship_id}")
        return {"message": "Friendship deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete friendship error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Permission management endpoints
@router.post("/permissions", response_model=PermissionResponse)
async def create_permission(
    permission_data: PermissionCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Create or update permissions for a friend."""
    try:
        user_id = current_user["uid"]
        friend_id = permission_data.friend_id
        
        # Check if friendship exists and is accepted
        friendship = db.query(Friendship).filter(
            ((Friendship.user_id == user_id) & (Friendship.friend_id == friend_id)) |
            ((Friendship.user_id == friend_id) & (Friendship.friend_id == user_id)),
            Friendship.status == 'accepted'
        ).first()
        
        if not friendship:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Accepted friendship not found"
            )
        
        # Check if permission already exists
        existing_permission = db.query(Permission).filter(
            Permission.user_id == user_id,
            Permission.friend_id == friend_id
        ).first()
        
        if existing_permission:
            # Update existing permission
            existing_permission.auto_accept_reminders = permission_data.auto_accept_reminders
            existing_permission.auto_accept_notes = permission_data.auto_accept_notes
            existing_permission.updated_at = datetime.utcnow()
            permission = existing_permission
        else:
            # Create new permission
            permission = Permission(
                id=str(uuid.uuid4()),
                user_id=user_id,
                friend_id=friend_id,
                auto_accept_reminders=permission_data.auto_accept_reminders,
                auto_accept_notes=permission_data.auto_accept_notes
            )
            db.add(permission)
        
        db.commit()
        db.refresh(permission)
        
        # Get friend info
        friend = db.query(User).filter(User.id == friend_id).first()
        friend_info = None
        if friend:
            friend_info = UserBasicResponse(
                id=str(friend.id),
                email=friend.email,
                language=friend.language,
                timezone=friend.timezone
            )
        
        logger.info(f"Created/updated permission: {permission.id} for {user_id} -> {friend_id}")
        
        return PermissionResponse(
            id=str(permission.id),
            user_id=str(permission.user_id),
            friend_id=str(permission.friend_id),
            auto_accept_reminders=permission.auto_accept_reminders,
            auto_accept_notes=permission.auto_accept_notes,
            updated_at=permission.updated_at,
            friend_info=friend_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Create permission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/permissions", response_model=List[PermissionResponse])
async def get_permissions(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all permissions for the current user."""
    try:
        user_id = current_user["uid"]
        
        permissions = db.query(Permission).filter(
            Permission.user_id == user_id
        ).order_by(Permission.updated_at.desc()).all()
        
        result = []
        for permission in permissions:
            friend = db.query(User).filter(User.id == permission.friend_id).first()
            
            friend_info = None
            if friend:
                friend_info = UserBasicResponse(
                    id=str(friend.id),
                    email=friend.email,
                    language=friend.language,
                    timezone=friend.timezone
                )
            
            result.append(PermissionResponse(
                id=str(permission.id),
                user_id=str(permission.user_id),
                friend_id=str(permission.friend_id),
                auto_accept_reminders=permission.auto_accept_reminders,
                auto_accept_notes=permission.auto_accept_notes,
                updated_at=permission.updated_at,
                friend_info=friend_info
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Get permissions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 