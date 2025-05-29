from typing import List
from decimal import Decimal
from sqlalchemy.orm import Session
from models.models import LedgerEntry
from connect_db import SessionLocal
from utils.logger import logger
import uuid

class LedgerService:
    """Service for managing expenses and financial tracking."""
    
    def __init__(self):
        pass
    
    async def add_expense(self, user_id: str, amount: float, description: str) -> dict:
        """Add a new expense."""
        db = SessionLocal()
        try:
            entry = LedgerEntry(
                id=str(uuid.uuid4()),
                user_id=user_id,
                amount=Decimal(str(amount)),
                contact_name=description,
                direction="owe"  # Default to "owe" for expenses
            )
            
            db.add(entry)
            db.commit()
            db.refresh(entry)
            
            logger.info(f"Created ledger entry {entry.id} for user {user_id}")
            return {
                "id": str(entry.id),
                "user_id": str(entry.user_id),
                "amount": float(entry.amount),
                "description": entry.contact_name,
                "direction": entry.direction,
                "created_at": entry.created_at
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating ledger entry: {e}")
            raise
        finally:
            db.close()
    
    async def get_user_expenses(self, user_id: str) -> List[dict]:
        """Get all expenses for a user."""
        db = SessionLocal()
        try:
            entries = db.query(LedgerEntry).filter(LedgerEntry.user_id == user_id).all()
            return [
                {
                    "id": str(e.id),
                    "user_id": str(e.user_id),
                    "amount": float(e.amount),
                    "description": e.contact_name,
                    "direction": e.direction,
                    "created_at": e.created_at
                }
                for e in entries
            ]
        except Exception as e:
            logger.error(f"Error getting ledger entries: {e}")
            return []
        finally:
            db.close() 