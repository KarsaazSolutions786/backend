from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel, Field
from ..database import get_db
from sqlalchemy.exc import IntegrityError
from ..models import LedgerEntry, Customer
from datetime import datetime

# Simple auth stub (to be replaced with real JWT parsing)

def get_current_customer_id() -> int:
    return 2

router = APIRouter(prefix="/ledger-entries", tags=["ledger_entries"])

# Schemas
class LedgerEntryCreate(BaseModel):
    friend_id: int = Field(..., gt=0)
    amount: float = Field(..., gt=0)
    ledger_direction_id: int = Field(..., gt=0)
    notes: Optional[str] = None

class LedgerEntryUpdate(BaseModel):
    friend_id: Optional[int] = Field(None, gt=0)
    amount: Optional[float] = Field(None, gt=0)
    ledger_direction_id: Optional[int] = Field(None, gt=0)
    notes: Optional[str] = None

class LedgerEntryResponse(BaseModel):
    id: int
    customer_id: int
    friend_id: int
    amount: float
    ledger_direction_id: int
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class LedgerSummary(BaseModel):
    friend_id: int
    total_amount: float

# Create
@router.post("/", response_model=LedgerEntryResponse, status_code=status.HTTP_201_CREATED)
def create_ledger_entry(entry: LedgerEntryCreate, db: Session = Depends(get_db)):
    customer_id = get_current_customer_id()

    # Validate foreign keys
    if not db.query(Customer).filter(Customer.id == entry.friend_id).first():
        raise HTTPException(status_code=400, detail="Invalid friend_id")

    new_entry = LedgerEntry(
        customer_id=customer_id,
        friend_id=entry.friend_id,
        amount=entry.amount,
        ledger_direction_id=entry.ledger_direction_id,
        notes=entry.notes,
    )
    try:
        db.add(new_entry)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Database integrity error: " + str(e.orig))
    db.refresh(new_entry)
    return new_entry

# List all for current customer
@router.get("/", response_model=List[LedgerEntryResponse])
def list_entries(db: Session = Depends(get_db)):
    customer_id = get_current_customer_id()
    entries = db.query(LedgerEntry).filter(LedgerEntry.customer_id == customer_id).order_by(LedgerEntry.created_at.desc()).all()
    return entries

# Get by id
@router.get("/{entry_id}", response_model=LedgerEntryResponse)
def get_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Ledger entry not found")
    customer_id = get_current_customer_id()
    if entry.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return entry

# Update
@router.put("/{entry_id}", response_model=LedgerEntryResponse)
def update_entry(entry_id: int, update: LedgerEntryUpdate, db: Session = Depends(get_db)):
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Ledger entry not found")
    customer_id = get_current_customer_id()
    if entry.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="Access denied")
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(entry, field, value)
    db.commit()
    db.refresh(entry)
    return entry

# Delete
@router.delete("/{entry_id}")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Ledger entry not found")
    customer_id = get_current_customer_id()
    if entry.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="Access denied")
    db.delete(entry)
    db.commit()
    return {"message": "Ledger entry deleted successfully"}

# Summary per friend
@router.get("/summary", response_model=List[LedgerSummary])
def ledger_summary(db: Session = Depends(get_db)):
    customer_id = get_current_customer_id()
    rows = (
        db.query(LedgerEntry.friend_id, func.sum(LedgerEntry.amount).label("total"))
        .filter(LedgerEntry.customer_id == customer_id)
        .group_by(LedgerEntry.friend_id)
        .all()
    )
    return [LedgerSummary(friend_id=row[0], total_amount=float(row[1] or 0)) for row in rows] 