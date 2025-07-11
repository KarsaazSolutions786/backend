from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Query
from sqlalchemy.orm import Session, joinedload
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
import logging
import secrets

from ..database import get_db
from ..models import Customer, CustomerPreference as CustomerPreferences, CustomerDevice
from ..schemas import (
    CustomerResponse, CustomerCreate, CustomerUpdate,
    CustomerPreferencesResponse, CustomerPreferencesUpdate,
    CustomerDeviceResponse, CustomerDeviceCreate, CustomerDeviceUpdate,
    CustomersListResponse, CustomerStats
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

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=CustomersListResponse)
async def get_customers(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get list of customers with pagination"""
    offset = (page - 1) * limit
    
    query = db.query(Customer)
    
    if search:
        query = query.filter(Customer.email.contains(search))
    
    if is_active is not None:
        query = query.filter(Customer.is_active == is_active)
    
    total = query.count()
    customers = query.offset(offset).limit(limit).all()
    
    return CustomersListResponse(
        customers=[CustomerResponse.from_orm(customer) for customer in customers],
        total=total,
        page=page,
        limit=limit,
        pages=(total + limit - 1) // limit
    )

@router.get("/search-by-name", response_model=CustomersListResponse)
async def search_customers_by_name(
    name: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Search customers by full_name or customer_name (case-insensitive, partial match)"""
    offset = (page - 1) * limit
    from ..models import CustomerProfile
    query = db.query(Customer).join(CustomerProfile).filter(
        (CustomerProfile.full_name.ilike(f"%{name}%")) |
        (CustomerProfile.user_name.ilike(f"%{name}%"))
    )
    total = query.count()
    customers = query.offset(offset).limit(limit).all()
    return CustomersListResponse(
        customers=[CustomerResponse.from_orm(customer) for customer in customers],
        total=total,
        page=page,
        limit=limit,
        pages=(total + limit - 1) // limit
    )

@router.get("/me", response_model=CustomerResponse)
async def get_current_customer_profile(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get current customer profile"""
    customer = db.query(Customer).filter(Customer.id == current_customer_id).first()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    return CustomerResponse.from_orm(customer)

@router.put("/me", response_model=CustomerResponse)
async def update_current_customer(
    customer_data: CustomerUpdate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Update current customer"""
    customer = db.query(Customer).filter(Customer.id == current_customer_id).first()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    update_data = customer_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(customer, field, value)
    
    customer.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(customer)
    
    return CustomerResponse.from_orm(customer)

@router.get("/stats", response_model=CustomerStats)
async def get_customer_stats(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get customer statistics"""
    total_customers = db.query(Customer).count()
    active_customers = db.query(Customer).filter(Customer.is_active == True).count()
    verified_customers = db.query(Customer).filter(Customer.is_verified == True).count()
    
    return CustomerStats(
        total_customers=total_customers,
        active_customers=active_customers,
        verified_customers=verified_customers,
        inactive_customers=total_customers - active_customers
    )

@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: int,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get customer by ID"""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    return CustomerResponse.from_orm(customer)

@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: int,
    customer_data: CustomerUpdate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Update customer by ID"""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    update_data = customer_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(customer, field, value)
    
    customer.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(customer)
    
    return CustomerResponse.from_orm(customer)

@router.delete("/{customer_id}")
async def delete_customer(
    customer_id: int,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Delete customer"""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    db.delete(customer)
    db.commit()
    
    return {"message": "Customer deleted successfully"}

@router.get("/me/preferences", response_model=CustomerPreferencesResponse)
async def get_customer_preferences(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get current customer preferences"""
    preferences = db.query(CustomerPreferences).filter(
        CustomerPreferences.customer_id == current_customer_id
    ).first()
    
    if not preferences:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer preferences not found"
        )
    
    return CustomerPreferencesResponse.from_orm(preferences)

@router.put("/me/preferences", response_model=CustomerPreferencesResponse)
async def update_customer_preferences(
    preferences_data: CustomerPreferencesUpdate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Update current customer preferences"""
    preferences = db.query(CustomerPreferences).filter(
        CustomerPreferences.customer_id == current_customer_id
    ).first()
    
    if not preferences:
        # Create preferences if they don't exist
        preferences = CustomerPreferences(
            customer_id=current_customer_id
        )
        db.add(preferences)
    
    update_data = preferences_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(preferences, field, value)
    
    preferences.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(preferences)
    
    return CustomerPreferencesResponse.from_orm(preferences)

@router.get("/me/devices", response_model=List[CustomerDeviceResponse])
async def get_customer_devices(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get current customer devices"""
    devices = db.query(CustomerDevice).filter(
        CustomerDevice.customer_id == current_customer_id,
        CustomerDevice.is_active == True
    ).all()
    
    return [CustomerDeviceResponse.from_orm(device) for device in devices]

@router.post("/me/devices", response_model=CustomerDeviceResponse)
async def register_customer_device(
    device_data: CustomerDeviceCreate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Register a new device for current customer"""
    device = CustomerDevice(
        customer_id=current_customer_id,
        **device_data.dict()
    )
    
    db.add(device)
    db.commit()
    db.refresh(device)
    
    return CustomerDeviceResponse.from_orm(device)

@router.delete("/me/devices/{device_id}")
async def unregister_customer_device(
    device_id: int,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Unregister a device for current customer"""
    device = db.query(CustomerDevice).filter(
        CustomerDevice.id == device_id,
        CustomerDevice.customer_id == current_customer_id
    ).first()
    
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    device.is_active = False
    db.commit()
    
    return {"message": "Device unregistered successfully"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "customer-service"} 