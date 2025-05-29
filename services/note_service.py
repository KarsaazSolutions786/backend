from typing import List
from sqlalchemy.orm import Session
from models.models import Note
from connect_db import SessionLocal
from utils.logger import logger
import uuid

class NoteService:
    """Service for managing notes."""
    
    def __init__(self):
        pass
    
    async def create_note(self, user_id: str, title: str, content: str) -> dict:
        """Create a new note."""
        db = SessionLocal()
        try:
            note = Note(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=f"{title}\n{content}" if title else content,
                source="service"
            )
            
            db.add(note)
            db.commit()
            db.refresh(note)
            
            logger.info(f"Created note {note.id} for user {user_id}")
            return {
                "id": str(note.id),
                "user_id": str(note.user_id),
            "title": title,
                "content": content,
                "created_at": note.created_at
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating note: {e}")
            raise
        finally:
            db.close()
    
    async def get_user_notes(self, user_id: str) -> List[dict]:
        """Get all notes for a user."""
        db = SessionLocal()
        try:
            notes = db.query(Note).filter(Note.user_id == user_id).all()
            return [
                {
                    "id": str(n.id),
                    "user_id": str(n.user_id),
                    "content": n.content,
                    "source": n.source,
                    "created_at": n.created_at
                }
                for n in notes
            ]
        except Exception as e:
            logger.error(f"Error getting notes: {e}")
            return []
        finally:
            db.close() 