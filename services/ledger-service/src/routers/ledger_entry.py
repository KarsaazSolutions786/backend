from fastapi import APIRouter, HTTPException, Depends, Query, Request, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel, Field
from ..database import get_db
from sqlalchemy.exc import IntegrityError
from ..models import LedgerEntry, Customer, Friendship
from datetime import datetime
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from fastapi.responses import JSONResponse
import os
# Use secure shared authentication 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../shared'))

try:
    from simple_auth import get_current_customer_id
except ImportError:
    # Fallback to local secure implementation
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi import Depends, HTTPException, status
    import jwt
    
    def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
        """Secure local implementation as fallback"""
        try:
            token = credentials.credentials
            secret_key = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
            
            # Validate production environment
            if os.getenv("ENVIRONMENT") == "production" and secret_key in [
                "your-secret-key-here", 
                "your-secret-key-here-change-in-production",
                "eindr-super-secret-key-change-in-production-123456789"
            ]:
                raise ValueError("Production environment requires a secure SECRET_KEY")
            
            payload = jwt.decode(
                token, 
                secret_key, 
                algorithms=["HS256"],
                audience="eindr-api",
                issuer="eindr-issuer",
                options={"verify_signature": True, "verify_exp": True, "verify_aud": True, "verify_iss": True}
            )
            customer_id = payload.get("sub")
            if customer_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing subject"
                )
            return int(customer_id)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

# Create local limiter instance
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/ledger-entries", tags=["ledger_entries"])

# Schemas
class LedgerEntryCreate(BaseModel):
    # For app users (friends) - either friend_id OR friend contact info required
    friend_id: Optional[int] = Field(None, gt=0)
    
    # For non-app contacts
    friend_name: Optional[str] = Field(None, min_length=1, max_length=255)
    friend_phone: Optional[str] = Field(None, min_length=1, max_length=20)
    friend_email: Optional[str] = Field(None, min_length=1, max_length=255)
    
    # Transaction details
    amount: float = Field(..., gt=0)
    ledger_direction_id: int = Field(..., gt=0)
    notes: Optional[str] = None
    status: str = Field(default="saved", pattern="^(draft|saved)$")
    
    def validate_friend_info(self):
        """Ensure either friend_id or friend contact info is provided"""
        if not self.friend_id and not (self.friend_name or self.friend_phone or self.friend_email):
            raise ValueError("Either friend_id or friend contact information (name, phone, or email) must be provided")
        if self.friend_id and (self.friend_name or self.friend_phone or self.friend_email):
            raise ValueError("Cannot specify both friend_id and friend contact information")
        return self

class LedgerEntryUpdate(BaseModel):
    # For app users (friends)
    friend_id: Optional[int] = Field(None, gt=0)
    
    # For non-app contacts
    friend_name: Optional[str] = Field(None, min_length=1, max_length=255)
    friend_phone: Optional[str] = Field(None, min_length=1, max_length=20)
    friend_email: Optional[str] = Field(None, min_length=1, max_length=255)
    
    # Transaction details
    amount: Optional[float] = Field(None, gt=0)
    ledger_direction_id: Optional[int] = Field(None, gt=0)
    notes: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(draft|saved)$")

class LedgerEntryResponse(BaseModel):
    id: int
    customer_id: int
    
    # For app users (friends)
    friend_id: Optional[int]
    
    # For non-app contacts
    friend_name: Optional[str]
    friend_phone: Optional[str]
    friend_email: Optional[str]
    
    # Transaction details
    amount: float
    ledger_direction_id: int
    notes: Optional[str]
    status: str
    
    # Metadata
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class LedgerSummary(BaseModel):
    friend_id: Optional[int]
    friend_name: Optional[str]
    friend_phone: Optional[str]
    friend_email: Optional[str]
    total_amount: float
    transaction_count: int
    is_app_user: bool

class LedgerSummaryResponse(BaseModel):
    app_friends: List[LedgerSummary]
    non_app_contacts: List[LedgerSummary]
    total_balance: float
    total_transactions: int

class FriendInfo(BaseModel):
    id: int
    customer_id: int
    friend_id: int
    friend_name: str
    friend_email: str
    status: str

# Create
@router.post("/", response_model=LedgerEntryResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
def create_ledger_entry(request: Request, entry: LedgerEntryCreate, db: Session = Depends(get_db), current_customer_id: int = Depends(get_current_customer_id)):
    customer_id = current_customer_id

    # Validate friend information
    try:
        entry.validate_friend_info()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # If friend_id is provided, validate it exists
    if entry.friend_id:
        friend = db.query(Customer).filter(Customer.id == entry.friend_id).first()
        if not friend:
            raise HTTPException(status_code=400, detail="Invalid friend_id: Friend not found")

    # Create new ledger entry
    new_entry = LedgerEntry(
        customer_id=customer_id,
        friend_id=entry.friend_id,
        friend_name=entry.friend_name,
        friend_phone=entry.friend_phone,
        friend_email=entry.friend_email,
        amount=entry.amount,
        ledger_direction_id=entry.ledger_direction_id,
        notes=entry.notes,
        status=entry.status,
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
def list_entries(db: Session = Depends(get_db), current_customer_id: int = Depends(get_current_customer_id)):
    customer_id = current_customer_id
    entries = db.query(LedgerEntry).filter(LedgerEntry.customer_id == customer_id).order_by(LedgerEntry.created_at.desc()).all()
    return entries

# Get friends list for dropdown - MUST be before parameterized routes
@router.get("/friends", response_model=List[FriendInfo])
async def get_friends_list(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get list of accepted friends for dropdown selection using direct database query"""
    try:
        # Query for friendships involving the current user with accepted status only
        accepted_friendships = db.query(Friendship).filter(
            (Friendship.user_id == current_customer_id) | (Friendship.friend_id == current_customer_id),
            Friendship.status == "accepted"
        ).order_by(Friendship.created_at.desc()).all()
        
        friends_list = []
        for friendship in accepted_friendships:
            # Determine the friend's details
            if friendship.user_id == current_customer_id:
                friend_id = friendship.friend_id
            else:
                friend_id = friendship.user_id
            
            # Get friend's customer details
            friend_customer = db.query(Customer).filter(Customer.id == friend_id).first()
            if friend_customer:
                friends_list.append(FriendInfo(
                    id=friendship.id,
                    customer_id=current_customer_id,
                    friend_id=friend_id,
                    friend_name=friend_customer.email.split('@')[0],  # Use email prefix as name
                    friend_email=friend_customer.email,
                    status=friendship.status
                ))
        
        return friends_list
        
    except Exception as e:
        print(f"Error fetching friends list: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch friends list"
        )

# Get by id
@router.get("/{entry_id}", response_model=LedgerEntryResponse)
def get_entry(entry_id: int, db: Session = Depends(get_db), current_customer_id: int = Depends(get_current_customer_id)):
    customer_id = current_customer_id
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Ledger entry not found")
    if entry.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return entry

# Update
@router.put("/{entry_id}", response_model=LedgerEntryResponse)
def update_entry(entry_id: int, update: LedgerEntryUpdate, db: Session = Depends(get_db), current_customer_id: int = Depends(get_current_customer_id)):
    customer_id = current_customer_id
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Ledger entry not found")
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
def delete_entry(entry_id: int, db: Session = Depends(get_db), current_customer_id: int = Depends(get_current_customer_id)):
    customer_id = current_customer_id
    entry = db.query(LedgerEntry).filter(LedgerEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Ledger entry not found")
    if entry.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="Access denied")
    db.delete(entry)
    db.commit()
    return {"message": "Ledger entry deleted successfully"}

# Enhanced summary endpoint for both app friends and non-app contacts
@router.get("/summary", response_model=LedgerSummaryResponse)
def ledger_summary(db: Session = Depends(get_db), current_customer_id: int = Depends(get_current_customer_id)):
    customer_id = current_customer_id
    
    # Get summaries for app friends (where friend_id is not null)
    app_friends_data = (
        db.query(
            LedgerEntry.friend_id,
            func.sum(LedgerEntry.amount).label("total_amount"),
            func.count(LedgerEntry.id).label("transaction_count")
        )
        .filter(LedgerEntry.customer_id == customer_id)
        .filter(LedgerEntry.friend_id.isnot(None))
        .group_by(LedgerEntry.friend_id)
        .all()
    )
    
    # Get summaries for non-app contacts (where friend_id is null)
    non_app_contacts_data = (
        db.query(
            LedgerEntry.friend_name,
            LedgerEntry.friend_phone,
            LedgerEntry.friend_email,
            func.sum(LedgerEntry.amount).label("total_amount"),
            func.count(LedgerEntry.id).label("transaction_count")
        )
        .filter(LedgerEntry.customer_id == customer_id)
        .filter(LedgerEntry.friend_id.is_(None))
        .group_by(LedgerEntry.friend_name, LedgerEntry.friend_phone, LedgerEntry.friend_email)
        .all()
    )
    
    # Build app friends summary
    app_friends = [
        LedgerSummary(
            friend_id=row[0],
            friend_name=None,
            friend_phone=None,
            friend_email=None,
            total_amount=float(row[1] or 0),
            transaction_count=int(row[2] or 0),
            is_app_user=True
        )
        for row in app_friends_data
    ]
    
    # Build non-app contacts summary
    non_app_contacts = [
        LedgerSummary(
            friend_id=None,
            friend_name=row[0],
            friend_phone=row[1],
            friend_email=row[2],
            total_amount=float(row[3] or 0),
            transaction_count=int(row[4] or 0),
            is_app_user=False
        )
        for row in non_app_contacts_data
    ]
    
    # Calculate totals
    total_balance = sum([friend.total_amount for friend in app_friends + non_app_contacts])
    total_transactions = sum([friend.transaction_count for friend in app_friends + non_app_contacts])
    
    return LedgerSummaryResponse(
        app_friends=app_friends,
        non_app_contacts=non_app_contacts,
        total_balance=total_balance,
         total_transactions=total_transactions
     )