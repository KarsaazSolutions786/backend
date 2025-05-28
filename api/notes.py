from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.models import Note
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger
import uuid

router = APIRouter()

# Pydantic models
class NoteCreate(BaseModel):
    content: Optional[str] = None
    source: Optional[str] = None

class NoteUpdate(BaseModel):
    content: Optional[str] = None
    source: Optional[str] = None

class NoteResponse(BaseModel):
    id: str
    user_id: str
    content: Optional[str]
    source: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

@router.post("/", response_model=NoteResponse)
async def create_note(
    note_data: NoteCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Create a new note."""
    try:
        user_id = current_user["uid"]
        
        note = Note(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=note_data.content,
            source=note_data.source
        )
        
        db.add(note)
        db.commit()
        db.refresh(note)
        
        logger.info(f"Created note: {note.id} for user: {user_id}")
        return NoteResponse(
            id=str(note.id),
            user_id=str(note.user_id),
            content=note.content,
            source=note.source,
            created_at=note.created_at
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Create note error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[NoteResponse])
async def get_notes(
    source: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all notes for the current user."""
    try:
        user_id = current_user["uid"]
        query = db.query(Note).filter(Note.user_id == user_id)
        
        if source:
            query = query.filter(Note.source == source)
        
        notes = query.order_by(Note.created_at.desc()).all()
        
        return [
            NoteResponse(
                id=str(note.id),
                user_id=str(note.user_id),
                content=note.content,
                source=note.source,
                created_at=note.created_at
            )
            for note in notes
        ]
        
    except Exception as e:
        logger.error(f"Get notes error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get a specific note."""
    try:
        user_id = current_user["uid"]
        note = db.query(Note).filter(
            Note.id == note_id,
            Note.user_id == user_id
        ).first()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        return NoteResponse(
            id=str(note.id),
            user_id=str(note.user_id),
            content=note.content,
            source=note.source,
            created_at=note.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get note error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str,
    note_data: NoteUpdate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Update a note."""
    try:
        user_id = current_user["uid"]
        note = db.query(Note).filter(
            Note.id == note_id,
            Note.user_id == user_id
        ).first()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # Update fields if provided
        update_data = note_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(note, field, value)
        
        db.commit()
        db.refresh(note)
        
        logger.info(f"Updated note: {note_id}")
        return NoteResponse(
            id=str(note.id),
            user_id=str(note.user_id),
            content=note.content,
            source=note.source,
            created_at=note.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update note error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{note_id}")
async def delete_note(
    note_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Delete a note."""
    try:
        user_id = current_user["uid"]
        note = db.query(Note).filter(
            Note.id == note_id,
            Note.user_id == user_id
        ).first()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        db.delete(note)
        db.commit()
        
        logger.info(f"Deleted note: {note_id}")
        return {"message": "Note deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete note error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/search/{query}", response_model=List[NoteResponse])
async def search_notes(
    query: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Search notes by content."""
    try:
        user_id = current_user["uid"]
        
        notes = db.query(Note).filter(
            Note.user_id == user_id,
            Note.content.ilike(f"%{query}%")
        ).order_by(Note.created_at.desc()).all()
        
        return [
            NoteResponse(
                id=str(note.id),
                user_id=str(note.user_id),
                content=note.content,
                source=note.source,
                created_at=note.created_at
            )
            for note in notes
        ]
        
    except Exception as e:
        logger.error(f"Search notes error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 