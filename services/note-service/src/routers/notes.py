from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, desc, func, text
from typing import List, Optional
from datetime import datetime
import logging

from ..database import get_db
from ..models import Note, NoteShare, Customer  
from ..schemas import (
    NoteCreate, NoteUpdate, NoteResponse, NoteWithShares,
    NoteShareCreate, NoteShareUpdate, NoteShareResponse,
    NotesListResponse, NoteFilters, NoteBulkUpdate, NoteBulkDelete,
    NoteStats, ErrorResponse
)
# from shared.simple_auth import get_current_customer_id  # Temporarily disabled

# Temporary local implementation
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
import jwt
import os

def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
    """Temporary local implementation of get_current_customer_id"""
    try:
        token = credentials.credentials
        secret_key = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        customer_id = payload.get("sub")
        if customer_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return int(customer_id)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.responses import JSONResponse

# Create local limiter instance
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/notes", tags=["Notes"])
security = HTTPBearer()

logger = logging.getLogger(__name__)

@router.post("/", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_note(
    request: Request,
    note_data: NoteCreate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Create a new note"""
    new_note = Note(
        customer_id=current_customer_id,
        title=note_data.title,
        description=note_data.description,
        content_type=note_data.content_type.value if note_data.content_type else "text",
        is_favorite=note_data.is_favorite,
        is_pinned=note_data.is_pinned
    )
    
    db.add(new_note)
    db.commit()
    db.refresh(new_note)
    
    # Load customer relationship
    note = db.query(Note).options(joinedload(Note.customer)).filter(Note.id == new_note.id).first()
    
    return NoteResponse.from_orm(note)

@router.get("/", response_model=NotesListResponse)
async def get_notes(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    is_favorite: Optional[bool] = Query(None),
    is_pinned: Optional[bool] = Query(None),
    is_shared: Optional[bool] = Query(None),
    content_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    shared_with_me: Optional[bool] = Query(None),
    sort_by: str = Query("created_at", regex="^(created_at|updated_at|title|last_accessed)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get notes with filtering, pagination, and sorting"""
    customer_id = current_customer_id
    offset = (page - 1) * limit
    
    if shared_with_me:
        # Get notes shared with this customer
        query = db.query(Note).join(
            NoteShare, Note.id == NoteShare.note_id
        ).options(joinedload(Note.customer)).filter(
            and_(
                NoteShare.shared_with_id == customer_id,
                NoteShare.is_active == True,
                or_(
                    NoteShare.expires_at.is_(None),
                    NoteShare.expires_at > datetime.utcnow()
                )
            )
        )
    else:
        # Get customer's own notes
        query = db.query(Note).options(joinedload(Note.customer)).filter(
            Note.customer_id == customer_id
        )
        
        # Apply filters
    if is_favorite is not None:
        query = query.filter(Note.is_favorite == is_favorite)
        
    if is_pinned is not None:
        query = query.filter(Note.is_pinned == is_pinned)
    
    if is_shared is not None:
        query = query.filter(Note.is_shared == is_shared)
    
    if content_type:
        query = query.filter(Note.content_type == content_type)
        
        if search:
            search_filter = or_(
                Note.title.ilike(f"%{search}%"),
                Note.description.ilike(f"%{search}%")
            )
            query = query.filter(search_filter)
    
    # Apply sorting
    if sort_order == "desc":
        query = query.order_by(desc(getattr(Note, sort_by)))
    else:
        query = query.order_by(getattr(Note, sort_by))
    
    # Get total count
    total = query.count()
        
        # Apply pagination
    notes = query.offset(offset).limit(limit).all()
    
    return NotesListResponse(
        notes=[NoteResponse.from_orm(note) for note in notes],
        total=total,
        page=page,
        limit=limit,
        has_next=(page * limit) < total,
        has_prev=page > 1
        )

@router.get("/stats", response_model=NoteStats)
async def get_note_stats(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get note statistics for current customer"""
    customer_id = current_customer_id
    
    # Basic counts
    total_notes = db.query(Note).filter(Note.customer_id == customer_id).count()
    favorite_notes = db.query(Note).filter(
        and_(Note.customer_id == customer_id, Note.is_favorite == True)
    ).count()
    pinned_notes = db.query(Note).filter(
        and_(Note.customer_id == customer_id, Note.is_pinned == True)
    ).count()
    shared_notes = db.query(Note).filter(
        and_(Note.customer_id == customer_id, Note.is_shared == True)
    ).count()
    
    # Notes shared with me
    notes_shared_with_me = db.query(NoteShare).filter(
        and_(
            NoteShare.shared_with_id == customer_id,
            NoteShare.status == "accepted"
        )
    ).count()
        
    # Notes by content type
    content_type_stats = db.query(
        Note.content_type,
        func.count(Note.id).label('count')
    ).filter(Note.customer_id == customer_id).group_by(Note.content_type).all()
    
    notes_by_content_type = {stat.content_type: stat.count for stat in content_type_stats}
    
    return NoteStats(
        total_notes=total_notes,
        favorite_notes=favorite_notes,
        pinned_notes=pinned_notes,
        shared_notes=shared_notes,
        notes_shared_with_me=notes_shared_with_me,
        notes_by_content_type=notes_by_content_type
    )

@router.get("/{note_id}", response_model=NoteWithShares)
async def get_note(
    note_id: int,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get a specific note with sharing information"""
    note = db.query(Note).options(
        joinedload(Note.customer),
        joinedload(Note.shares).joinedload(NoteShare.shared_by),
        joinedload(Note.shares).joinedload(NoteShare.shared_with)
    ).filter(Note.id == note_id).first()
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
        
    # Check if customer has access to this note
    has_access = (
        note.customer_id == current_customer_id or
        any(
            share.shared_with_id == current_customer_id and 
            share.is_active and 
            (share.expires_at is None or share.expires_at > datetime.utcnow())
            for share in note.shares
        )
    )
    
    if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
    # Update last_accessed if it's the note owner
    if note.customer_id == current_customer_id:
        note.last_accessed = datetime.utcnow()
        db.commit()
    
    return NoteWithShares.from_orm(note)

@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: int,
    note_data: NoteUpdate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Update a note"""
    note = db.query(Note).filter(Note.id == note_id).first()
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    # Check if customer can edit this note
    can_edit = note.customer_id == current_customer_id
    
    if not can_edit:
        # Check if shared with edit permission
        share = db.query(NoteShare).filter(
            and_(
                NoteShare.note_id == note_id,
                NoteShare.shared_with_id == current_customer_id,
                NoteShare.can_edit == True,
                NoteShare.is_active == True,
                or_(
                    NoteShare.expires_at.is_(None),
                    NoteShare.expires_at > datetime.utcnow()
                )
            )
        ).first()
        can_edit = share is not None
    
    if not can_edit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to edit this note"
        )
    
    # Update fields
    update_data = note_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field == "content_type" and value:
            setattr(note, field, value.value)
        else:
            setattr(note, field, value)
    
    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    
    # Load customer relationship
    note = db.query(Note).options(joinedload(Note.customer)).filter(Note.id == note_id).first()
    
    return NoteResponse.from_orm(note)

@router.delete("/{note_id}")
async def delete_note(
    note_id: int,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Delete a note"""
    note = db.query(Note).filter(Note.id == note_id).first()
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    # Check if customer can delete this note
    can_delete = note.customer_id == current_customer_id
    
    if not can_delete:
        # Check if shared with delete permission
        share = db.query(NoteShare).filter(
            and_(
                NoteShare.note_id == note_id,
                NoteShare.shared_with_id == current_customer_id,
                NoteShare.can_delete == True,
                NoteShare.is_active == True,
                or_(
                    NoteShare.expires_at.is_(None),
                    NoteShare.expires_at > datetime.utcnow()
                )
            )
        ).first()
        can_delete = share is not None
    
    if not can_delete:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this note"
        )
    
    db.delete(note)
    db.commit()
    
    return {"message": "Note deleted successfully"}

@router.post("/{note_id}/share", response_model=NoteShareResponse, status_code=status.HTTP_201_CREATED)
async def share_note(
    note_id: int,
    share_data: NoteShareCreate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Share a note with another customer"""
    note = db.query(Note).filter(Note.id == note_id).first()
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
    
    # Check if customer owns this note or can reshare
    can_share = note.customer_id == current_customer_id
    
    if not can_share:
        share = db.query(NoteShare).filter(
            and_(
                NoteShare.note_id == note_id,
                NoteShare.shared_with_id == current_customer_id,
                NoteShare.can_reshare == True,
                NoteShare.is_active == True,
                or_(
                    NoteShare.expires_at.is_(None),
                    NoteShare.expires_at > datetime.utcnow()
                )
            )
        ).first()
        can_share = share is not None
    
    if not can_share:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to share this note"
        )
    
    # Check if customer exists
    shared_with_customer = db.query(Customer).filter(Customer.id == share_data.shared_with_id).first()
    if not shared_with_customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer to share with not found"
        )
    
    # Check if already shared
    existing_share = db.query(NoteShare).filter(
        and_(
            NoteShare.note_id == note_id,
            NoteShare.shared_with_id == share_data.shared_with_id,
            NoteShare.is_active == True
        )
    ).first()
    
    if existing_share:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Note is already shared with this customer"
        )
    
    new_share = NoteShare(
        note_id=note_id,
        shared_by_id=current_customer_id,
        shared_with_id=share_data.shared_with_id,
        permission_level=share_data.permission_level.value,
        can_edit=share_data.can_edit,
        can_delete=share_data.can_delete,
        can_reshare=share_data.can_reshare,
        expires_at=share_data.expires_at
    )
    
    db.add(new_share)
    
    # Update note's is_shared flag
    note.is_shared = True
    
    db.commit()
    db.refresh(new_share)
    
    # Load relationships
    share = db.query(NoteShare).options(
        joinedload(NoteShare.shared_by),
        joinedload(NoteShare.shared_with)
    ).filter(NoteShare.id == new_share.id).first()
    
    return NoteShareResponse.from_orm(share)

@router.get("/{note_id}/shares", response_model=List[NoteShareResponse])
async def get_note_shares(
    note_id: int,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get all shares for a note"""
    note = db.query(Note).filter(Note.id == note_id).first()
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found"
        )
        
    # Check if customer has access to this note
    has_access = (
        note.customer_id == current_customer_id or
        db.query(NoteShare).filter(
            and_(
                NoteShare.note_id == note_id,
                NoteShare.shared_with_id == current_customer_id,
                NoteShare.is_active == True,
                or_(
                    NoteShare.expires_at.is_(None),
                    NoteShare.expires_at > datetime.utcnow()
                )
            )
        ).first() is not None
    )
    
    if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
    shares = db.query(NoteShare).options(
        joinedload(NoteShare.shared_by),
        joinedload(NoteShare.shared_with)
    ).filter(
        and_(
            NoteShare.note_id == note_id,
            NoteShare.is_active == True
        )
    ).all()
        
    return [NoteShareResponse.from_orm(share) for share in shares]

@router.put("/shares/{share_id}", response_model=NoteShareResponse)
async def update_note_share(
    share_id: int,
    share_data: NoteShareUpdate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Update note sharing permissions"""
    share = db.query(NoteShare).options(
        joinedload(NoteShare.note),
        joinedload(NoteShare.shared_by),
        joinedload(NoteShare.shared_with)
    ).filter(NoteShare.id == share_id).first()
    
    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share not found"
        )

    # Only note owner or share creator can update
    if share.note.customer_id != current_customer_id and share.shared_by_id != current_customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to update this share"
        )
    
    # Update fields
    update_data = share_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field == "permission_level" and value:
            setattr(share, field, value.value)
        else:
            setattr(share, field, value)
    
    db.commit()
    db.refresh(share)
    
    return NoteShareResponse.from_orm(share)

@router.delete("/shares/{share_id}")
async def revoke_note_share(
    share_id: int,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Revoke note sharing"""
    share = db.query(NoteShare).options(joinedload(NoteShare.note)).filter(NoteShare.id == share_id).first()
    
    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share not found"
        )
    
    # Note owner, share creator, or recipient can revoke
    can_revoke = (
        share.note.customer_id == current_customer_id or
        share.shared_by_id == current_customer_id or
        share.shared_with_id == current_customer_id
    )
    
    if not can_revoke:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to revoke this share"
        )

    db.delete(share)
    
    # Update note's is_shared flag if no more active shares
    remaining_shares = db.query(NoteShare).filter(
        and_(
            NoteShare.note_id == share.note_id,
            NoteShare.is_active == True,
            NoteShare.id != share_id
        )
    ).count()
    
    if remaining_shares == 0:
        share.note.is_shared = False
    
    db.commit()
    
    return {"message": "Share revoked successfully"}

@router.post("/bulk-update")
async def bulk_update_notes(
    bulk_data: NoteBulkUpdate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Bulk update notes (favorite, pinned status)"""
    # Verify all notes belong to current customer
    notes = db.query(Note).filter(
        and_(
            Note.id.in_(bulk_data.note_ids),
            Note.customer_id == current_customer_id
        )
    ).all()
    
    if len(notes) != len(bulk_data.note_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Some notes not found or access denied"
        )
    
    # Update notes
    update_data = bulk_data.dict(exclude_unset=True, exclude={"note_ids"})
    if update_data:
        for note in notes:
            for field, value in update_data.items():
                setattr(note, field, value)
            note.updated_at = datetime.utcnow()
        
        db.commit()
    
    return {"message": f"Successfully updated {len(notes)} notes"}

@router.delete("/bulk-delete")
async def bulk_delete_notes(
    bulk_data: NoteBulkDelete,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Bulk delete notes"""
    # Verify all notes belong to current customer
    notes = db.query(Note).filter(
        and_(
            Note.id.in_(bulk_data.note_ids),
            Note.customer_id == current_customer_id
        )
    ).all()
    
    if len(notes) != len(bulk_data.note_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Some notes not found or access denied"
        )
    
    # Delete notes
    for note in notes:
        db.delete(note)
    
    db.commit()
    
    return {"message": f"Successfully deleted {len(notes)} notes"}
