from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from decimal import Decimal
from sqlalchemy.orm import Session
from models.models import LedgerEntry
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger
import uuid

router = APIRouter()

# Pydantic models
class LedgerEntryCreate(BaseModel):
    contact_name: Optional[str] = None
    amount: Decimal
    direction: str  # 'owe' or 'owed'

    class Config:
        validate_assignment = True

class LedgerEntryUpdate(BaseModel):
    contact_name: Optional[str] = None
    amount: Optional[Decimal] = None
    direction: Optional[str] = None

    class Config:
        validate_assignment = True

class LedgerEntryResponse(BaseModel):
    id: str
    user_id: str
    contact_name: Optional[str]
    amount: Decimal
    direction: str
    created_at: datetime

    class Config:
        from_attributes = True

@router.post("/", response_model=LedgerEntryResponse)
async def create_ledger_entry(
    entry_data: LedgerEntryCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Create a new ledger entry."""
    try:
        # Validate direction
        if entry_data.direction not in ['owe', 'owed']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Direction must be either 'owe' or 'owed'"
            )
        
        user_id = current_user["uid"]
        
        entry = LedgerEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            contact_name=entry_data.contact_name,
            amount=entry_data.amount,
            direction=entry_data.direction
        )
        
        db.add(entry)
        db.commit()
        db.refresh(entry)
        
        logger.info(f"Created ledger entry: {entry.id} for user: {user_id}")
        return LedgerEntryResponse(
            id=str(entry.id),
            user_id=str(entry.user_id),
            contact_name=entry.contact_name,
            amount=entry.amount,
            direction=entry.direction,
            created_at=entry.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Create ledger entry error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[LedgerEntryResponse])
async def get_ledger_entries(
    direction: Optional[str] = None,
    contact_name: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all ledger entries for the current user."""
    try:
        user_id = current_user["uid"]
        query = db.query(LedgerEntry).filter(LedgerEntry.user_id == user_id)
        
        if direction:
            if direction not in ['owe', 'owed']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Direction must be either 'owe' or 'owed'"
                )
            query = query.filter(LedgerEntry.direction == direction)
        
        if contact_name:
            query = query.filter(LedgerEntry.contact_name.ilike(f"%{contact_name}%"))
        
        entries = query.order_by(LedgerEntry.created_at.desc()).all()
        
        return [
            LedgerEntryResponse(
                id=str(entry.id),
                user_id=str(entry.user_id),
                contact_name=entry.contact_name,
                amount=entry.amount,
                direction=entry.direction,
                created_at=entry.created_at
            )
            for entry in entries
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get ledger entries error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{entry_id}", response_model=LedgerEntryResponse)
async def get_ledger_entry(
    entry_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get a specific ledger entry."""
    try:
        user_id = current_user["uid"]
        entry = db.query(LedgerEntry).filter(
            LedgerEntry.id == entry_id,
            LedgerEntry.user_id == user_id
        ).first()
        
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ledger entry not found"
            )
        
        return LedgerEntryResponse(
            id=str(entry.id),
            user_id=str(entry.user_id),
            contact_name=entry.contact_name,
            amount=entry.amount,
            direction=entry.direction,
            created_at=entry.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get ledger entry error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{entry_id}", response_model=LedgerEntryResponse)
async def update_ledger_entry(
    entry_id: str,
    entry_data: LedgerEntryUpdate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Update a ledger entry."""
    try:
        # Validate direction if provided
        if entry_data.direction and entry_data.direction not in ['owe', 'owed']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Direction must be either 'owe' or 'owed'"
            )
        
        user_id = current_user["uid"]
        entry = db.query(LedgerEntry).filter(
            LedgerEntry.id == entry_id,
            LedgerEntry.user_id == user_id
        ).first()
        
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ledger entry not found"
            )
        
        # Update fields if provided
        update_data = entry_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(entry, field, value)
        
        db.commit()
        db.refresh(entry)
        
        logger.info(f"Updated ledger entry: {entry_id}")
        return LedgerEntryResponse(
            id=str(entry.id),
            user_id=str(entry.user_id),
            contact_name=entry.contact_name,
            amount=entry.amount,
            direction=entry.direction,
            created_at=entry.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update ledger entry error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{entry_id}")
async def delete_ledger_entry(
    entry_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Delete a ledger entry."""
    try:
        user_id = current_user["uid"]
        entry = db.query(LedgerEntry).filter(
            LedgerEntry.id == entry_id,
            LedgerEntry.user_id == user_id
        ).first()
        
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ledger entry not found"
            )
        
        db.delete(entry)
        db.commit()
        
        logger.info(f"Deleted ledger entry: {entry_id}")
        return {"message": "Ledger entry deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete ledger entry error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/summary/balance", response_model=dict)
async def get_balance_summary(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get balance summary (total owed and owed to user)."""
    try:
        user_id = current_user["uid"]
        
        # Calculate total amount user owes to others
        total_owe = db.query(LedgerEntry).filter(
            LedgerEntry.user_id == user_id,
            LedgerEntry.direction == 'owe'
        ).with_entities(db.func.sum(LedgerEntry.amount)).scalar() or 0
        
        # Calculate total amount owed to user
        total_owed = db.query(LedgerEntry).filter(
            LedgerEntry.user_id == user_id,
            LedgerEntry.direction == 'owed'
        ).with_entities(db.func.sum(LedgerEntry.amount)).scalar() or 0
        
        net_balance = total_owed - total_owe
        
        return {
            "total_owe": float(total_owe),
            "total_owed": float(total_owed),
            "net_balance": float(net_balance),
            "status": "positive" if net_balance > 0 else "negative" if net_balance < 0 else "balanced"
        }
        
    except Exception as e:
        logger.error(f"Get balance summary error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{ledger_id}/contact")
async def update_ledger_contact(
    ledger_id: str,
    contact_name: str,
    current_user: dict = Depends(verify_firebase_token)
):
    """Update the contact name for a specific ledger entry."""
    try:
        user_id = current_user["uid"]
        
        db = SessionLocal()
        try:
            # Find the ledger entry
            ledger_entry = db.query(LedgerEntry).filter(
                LedgerEntry.id == ledger_id,
                LedgerEntry.user_id == user_id
            ).first()
            
            if not ledger_entry:
                raise HTTPException(status_code=404, detail="Ledger entry not found")
            
            # Update the contact name
            old_name = ledger_entry.contact_name
            ledger_entry.contact_name = contact_name.strip()
            
            db.commit()
            
            logger.info(f"Updated ledger entry {ledger_id} contact from '{old_name}' to '{contact_name}'")
            
            return {
                "success": True,
                "message": "Contact name updated successfully",
                "data": {
                    "ledger_id": str(ledger_entry.id),
                    "old_contact_name": old_name,
                    "new_contact_name": contact_name,
                    "amount": float(ledger_entry.amount),
                    "direction": ledger_entry.direction
                }
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating ledger contact: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update contact name: {str(e)}"
        )

@router.put("/bulk-update-contacts")
async def bulk_update_unknown_contacts(
    contact_mapping: Dict[str, str],  # ledger_id -> contact_name
    current_user: dict = Depends(verify_firebase_token)
):
    """Bulk update contact names for multiple ledger entries."""
    try:
        user_id = current_user["uid"]
        
        db = SessionLocal()
        try:
            updated_entries = []
            
            for ledger_id, contact_name in contact_mapping.items():
                # Find and update each ledger entry
                ledger_entry = db.query(LedgerEntry).filter(
                    LedgerEntry.id == ledger_id,
                    LedgerEntry.user_id == user_id
                ).first()
                
                if ledger_entry:
                    old_name = ledger_entry.contact_name
                    ledger_entry.contact_name = contact_name.strip()
                    
                    updated_entries.append({
                        "ledger_id": str(ledger_entry.id),
                        "old_contact_name": old_name,
                        "new_contact_name": contact_name,
                        "amount": float(ledger_entry.amount)
                    })
            
            db.commit()
            
            logger.info(f"Bulk updated {len(updated_entries)} ledger entries for user {user_id}")
            
            return {
                "success": True,
                "message": f"Successfully updated {len(updated_entries)} ledger entries",
                "updated_count": len(updated_entries),
                "updated_entries": updated_entries
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk update: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to bulk update contacts: {str(e)}"
        ) 