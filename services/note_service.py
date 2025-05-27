from typing import List
from utils.logger import logger

class NoteService:
    """Service for managing notes."""
    
    def __init__(self):
        self.notes_db = {}
    
    async def create_note(self, user_id: str, title: str, content: str) -> dict:
        """Create a new note."""
        note_id = f"note_{len(self.notes_db) + 1}"
        note = {
            "id": note_id,
            "user_id": user_id,
            "title": title,
            "content": content
        }
        self.notes_db[note_id] = note
        return note 