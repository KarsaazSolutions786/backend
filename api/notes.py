from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.models import Note
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger

router = APIRouter()

# Pydantic models
class NoteCreate(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = []

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None

class NoteResponse(BaseModel):
    id: str
    title: str
    content: str
    tags: List[str]
    user_id: str
    created_at: datetime
    updated_at: datetime

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
        # Create new note
        note = Note(
            title=note_data.title,
            content=note_data.content,
            tags=",".join(note_data.tags) if note_data.tags else "",
            user_id=current_user["uid"]
        )
        
        db.add(note)
        db.commit()
        db.refresh(note)
        
        logger.info(f"Created note: {note.id} for user: {current_user['uid']}")
        return NoteResponse(
            **note.__dict__,
            tags=note.tags.split(",") if note.tags else []
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
    tag: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all notes for the current user."""
    try:
        query = db.query(Note).filter(Note.user_id == current_user["uid"])
        
        if tag:
            query = query.filter(Note.tags.contains(tag))
        
        notes = query.order_by(Note.updated_at.desc()).all()
        
        return [
            NoteResponse(
                **note.__dict__,
                tags=note.tags.split(",") if note.tags else []
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
        note = db.query(Note).filter(
            Note.id == note_id,
            Note.user_id == current_user["uid"]
        ).first()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        return NoteResponse(
            **note.__dict__,
            tags=note.tags.split(",") if note.tags else []
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
        note = db.query(Note).filter(
            Note.id == note_id,
            Note.user_id == current_user["uid"]
        ).first()
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # Update fields if provided
        if note_data.title is not None:
            note.title = note_data.title
        if note_data.content is not None:
            note.content = note_data.content
        if note_data.tags is not None:
            note.tags = ",".join(note_data.tags)
        
        db.commit()
        db.refresh(note)
        
        logger.info(f"Updated note: {note_id}")
        return NoteResponse(
            **note.__dict__,
            tags=note.tags.split(",") if note.tags else []
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
        note = db.query(Note).filter(
            Note.id == note_id,
            Note.user_id == current_user["uid"]
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