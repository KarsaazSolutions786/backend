from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models
class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    folder_id: Optional[str] = None
    tags: Optional[List[str]] = []
    category: Optional[str] = None
    is_shared: bool = False
    is_favorite: bool = False

class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = None
    folder_id: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    is_shared: Optional[bool] = None
    is_favorite: Optional[bool] = None

class NoteResponse(BaseModel):
    id: str
    user_id: str
    title: str
    content: str
    folder_id: Optional[str]
    tags: List[str]
    category: Optional[str]
    is_shared: bool
    is_favorite: bool
    word_count: int
    created_at: datetime
    updated_at: datetime

class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    color: Optional[str] = "#1f77b4"

class FolderResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    color: str
    note_count: int
    created_at: datetime

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    folder_id: Optional[str] = None

# Mock data storage (in production, use database)
notes_storage = {}
folders_storage = {}

def get_current_user_id() -> str:
    """Mock function to get current user ID"""
    return "user-123"

@router.post("/", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(note_data: NoteCreate):
    """Create a new note"""
    try:
        note_id = str(uuid.uuid4())
        user_id = get_current_user_id()
        
        # Calculate word count
        word_count = len(note_data.content.split()) if note_data.content else 0
        
        note = {
            "id": note_id,
            "user_id": user_id,
            "title": note_data.title,
            "content": note_data.content,
            "folder_id": note_data.folder_id,
            "tags": note_data.tags or [],
            "category": note_data.category,
            "is_shared": note_data.is_shared,
            "is_favorite": note_data.is_favorite,
            "word_count": word_count,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        notes_storage[note_id] = note
        
        logger.info(f"Created note: {note_id} for user: {user_id}")
        
        return NoteResponse(**note)
        
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create note"
        )

@router.get("/", response_model=List[NoteResponse])
async def get_notes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = Query(None),
    folder_id: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),  # Comma-separated tags
    is_favorite: Optional[bool] = Query(None),
    search: Optional[str] = Query(None)
):
    """Get user's notes with filtering options"""
    try:
        user_id = get_current_user_id()
        
        # Filter notes by user
        user_notes = [note for note in notes_storage.values() if note["user_id"] == user_id]
        
        # Apply filters
        if category:
            user_notes = [note for note in user_notes if note.get("category") == category]
        
        if folder_id:
            user_notes = [note for note in user_notes if note.get("folder_id") == folder_id]
        
        if is_favorite is not None:
            user_notes = [note for note in user_notes if note.get("is_favorite") == is_favorite]
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            user_notes = [note for note in user_notes 
                         if any(tag in note.get("tags", []) for tag in tag_list)]
        
        if search:
            search_lower = search.lower()
            user_notes = [note for note in user_notes 
                         if search_lower in note["title"].lower() or 
                            search_lower in note["content"].lower()]
        
        # Sort by updated_at (newest first)
        user_notes.sort(key=lambda x: x["updated_at"], reverse=True)
        
        # Apply pagination
        paginated_notes = user_notes[skip:skip + limit]
        
        return [NoteResponse(**note) for note in paginated_notes]
        
    except Exception as e:
        logger.error(f"Error getting notes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get notes"
        )

@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(note_id: str):
    """Get a specific note"""
    try:
        note = notes_storage.get(note_id)
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # Check if user owns the note or it's shared
        user_id = get_current_user_id()
        if note["user_id"] != user_id and not note.get("is_shared", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return NoteResponse(**note)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get note"
        )

@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(note_id: str, note_data: NoteUpdate):
    """Update a note"""
    try:
        note = notes_storage.get(note_id)
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # Check ownership
        user_id = get_current_user_id()
        if note["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Update fields
        update_data = note_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            note[field] = value
        
        # Update word count if content changed
        if "content" in update_data:
            note["word_count"] = len(note["content"].split()) if note["content"] else 0
        
        note["updated_at"] = datetime.utcnow()
        
        logger.info(f"Updated note: {note_id}")
        
        return NoteResponse(**note)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update note"
        )

@router.delete("/{note_id}")
async def delete_note(note_id: str):
    """Delete a note"""
    try:
        note = notes_storage.get(note_id)
        
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )
        
        # Check ownership
        user_id = get_current_user_id()
        if note["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        del notes_storage[note_id]
        
        logger.info(f"Deleted note: {note_id}")
        
        return {"message": "Note deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete note"
        )

@router.post("/search")
async def search_notes(search_data: SearchRequest):
    """Advanced note search"""
    try:
        user_id = get_current_user_id()
        user_notes = [note for note in notes_storage.values() if note["user_id"] == user_id]
        
        # Search in title and content
        query_lower = search_data.query.lower()
        matching_notes = []
        
        for note in user_notes:
            score = 0
            
            # Title match (higher weight)
            if query_lower in note["title"].lower():
                score += 10
            
            # Content match
            if query_lower in note["content"].lower():
                score += 5
            
            # Tag match
            if any(query_lower in tag.lower() for tag in note.get("tags", [])):
                score += 3
            
            if score > 0:
                matching_notes.append({"note": note, "score": score})
        
        # Sort by relevance score
        matching_notes.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "query": search_data.query,
            "results": [NoteResponse(**item["note"]) for item in matching_notes[:50]],
            "total_found": len(matching_notes)
        }
        
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search notes"
        )

@router.post("/folders", response_model=FolderResponse)
async def create_folder(folder_data: FolderCreate):
    """Create a new folder"""
    try:
        folder_id = str(uuid.uuid4())
        user_id = get_current_user_id()
        
        folder = {
            "id": folder_id,
            "user_id": user_id,
            "name": folder_data.name,
            "description": folder_data.description,
            "color": folder_data.color,
            "created_at": datetime.utcnow()
        }
        
        folders_storage[folder_id] = folder
        
        # Count notes in this folder
        note_count = sum(1 for note in notes_storage.values() 
                        if note.get("folder_id") == folder_id)
        
        return FolderResponse(**folder, note_count=note_count)
        
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create folder"
        )

@router.get("/folders", response_model=List[FolderResponse])
async def get_folders():
    """Get user's folders"""
    try:
        user_id = get_current_user_id()
        user_folders = [folder for folder in folders_storage.values() 
                       if folder["user_id"] == user_id]
        
        # Add note count for each folder
        result = []
        for folder in user_folders:
            note_count = sum(1 for note in notes_storage.values() 
                           if note.get("folder_id") == folder["id"])
            result.append(FolderResponse(**folder, note_count=note_count))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting folders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get folders"
        )

@router.get("/stats")
async def get_note_stats():
    """Get note statistics for the user"""
    try:
        user_id = get_current_user_id()
        user_notes = [note for note in notes_storage.values() if note["user_id"] == user_id]
        
        total_notes = len(user_notes)
        total_words = sum(note.get("word_count", 0) for note in user_notes)
        favorite_notes = sum(1 for note in user_notes if note.get("is_favorite", False))
        shared_notes = sum(1 for note in user_notes if note.get("is_shared", False))
        
        # Get all unique tags
        all_tags = set()
        for note in user_notes:
            all_tags.update(note.get("tags", []))
        
        return {
            "total_notes": total_notes,
            "total_words": total_words,
            "favorite_notes": favorite_notes,
            "shared_notes": shared_notes,
            "unique_tags": len(all_tags),
            "tags": list(all_tags)
        }
        
    except Exception as e:
        logger.error(f"Error getting note stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get note statistics"
        )
