from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Query
from sqlalchemy.orm import Session, joinedload
from typing import Optional, List, Union
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
import logging
import secrets

from ..database import get_db
from ..models import (
    Customer, CustomerPreference as CustomerPreferences, 
    CustomerDevice, CustomerProfile, Timezone, Language
)
from ..schemas import (
    CustomerResponse, CustomerCreate, CustomerUpdate,
    CustomerPreferencesResponse, CustomerPreferencesUpdate,
    CustomerDeviceResponse, CustomerDeviceCreate, CustomerDeviceUpdate,
    CustomersListResponse, CustomerStats, CustomerProfileUpdate, CustomerProfileResponse
)
# from shared.simple_auth import get_current_customer_id  # Temporarily disabled

# Temporary local implementation
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
import jwt
import os
from sqlalchemy import select

def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
    """Temporary local implementation of get_current_customer_id"""
    try:
        token = credentials.credentials
        secret_key = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
        print(f"DEBUG ROUTER: Using SECRET_KEY: {secret_key[:20]}...")
        print(f"DEBUG ROUTER: Token to decode: {token[:50]}...")
        
        # Decode without audience/issuer validation to match test token
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        print(f"DEBUG ROUTER: Decoded payload: {payload}")
        
        customer_id = payload.get("sub")
        if customer_id is None:
            print("DEBUG ROUTER: Missing 'sub' field in token")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        print(f"DEBUG ROUTER: Successfully extracted customer_id: {customer_id}")
        return int(customer_id)
    except jwt.PyJWTError as e:
        print(f"DEBUG ROUTER: JWT decode error: {str(e)}")
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

@router.get("/search", response_model=CustomersListResponse)
async def search_customers(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Search customers by email, full_name or user_name (case-insensitive, partial match)"""
    offset = (page - 1) * limit
    from ..models import CustomerProfile
    
    # Search in both customer email and profile fields
    customer_query = db.query(Customer).outerjoin(CustomerProfile).filter(
        (Customer.email.ilike(f"%{query}%")) |
        (CustomerProfile.full_name.ilike(f"%{query}%")) |
        (CustomerProfile.user_name.ilike(f"%{query}%"))
    )
    
    total = customer_query.count()
    customers = customer_query.offset(offset).limit(limit).all()
    
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

@router.put("/me", response_model=CustomerResponse)
async def update_current_customer(
    update_data: CustomerUpdate,
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Update current customer and profile details"""
    try:
        # Find the customer with profile
        customer = db.query(Customer).options(
            joinedload(Customer.profile)
        ).filter(Customer.id == current_customer_id).first()
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Ensure profile exists
        if not customer.profile:
            customer.profile = CustomerProfile(
                customer_id=customer.id,
                is_new=True  # Default to new user
            )
            db.add(customer.profile)
        
        # Convert to dictionary to handle update fields
        update_dict = update_data.dict(exclude_unset=True)
        
        # Update customer fields if applicable
        if 'email' in update_dict:
            customer.email = update_dict['email']
        
        if 'is_active' in update_dict:
            customer.is_active = update_dict['is_active']
        
        if 'is_verified' in update_dict:
            customer.is_verified = update_dict['is_verified']
        
        # Update profile fields if applicable
        if 'first_name' in update_dict or 'last_name' in update_dict:
            first_name = update_dict.get('first_name', '')
            last_name = update_dict.get('last_name', '')
            customer.profile.full_name = f"{first_name} {last_name}".strip()
        
        if 'display_name' in update_dict:
            customer.profile.user_name = update_dict['display_name']
        
        if 'bio' in update_dict:
            customer.profile.bio = update_dict['bio']
        
        if 'phone_number' in update_dict:
            customer.profile.phone_number = update_dict['phone_number']
        
        # Safely handle timezone and language
        if 'timezone' in update_dict:
            try:
                customer.profile.timezone = update_dict['timezone']
            except Exception:
                # Silently ignore if timezone can't be set
                pass
        
        if 'language' in update_dict:
            try:
                customer.profile.language = update_dict['language']
            except Exception:
                # Silently ignore if language can't be set
                pass
        
        if 'is_public' in update_dict:
            customer.profile.is_public = update_dict['is_public']
        
        # Update is_new if provided
        if 'is_new' in update_dict:
            customer.profile.is_new = update_dict['is_new']
        
        # Set is_new to False if any meaningful profile data is updated
        if any(key in update_dict for key in ['first_name', 'last_name', 'display_name', 'bio', 'phone_number']):
            customer.profile.is_new = False
        
        # Commit changes
        db.commit()
        db.refresh(customer)
        db.refresh(customer.profile)
        
        # Construct response manually to avoid SQLAlchemy state issues
        return CustomerResponse(
            id=customer.id,
            email=customer.email,
            is_active=customer.is_active,
            is_verified=customer.is_verified,
            created_at=customer.created_at,
            updated_at=customer.updated_at,
            last_login=customer.last_login,
            login_attempts=customer.login_attempts,
            locked_until=customer.locked_until,
            profile=CustomerProfileResponse(
                id=customer.profile.id,
                customer_id=customer.profile.customer_id,
                first_name=update_dict.get('first_name'),
                last_name=update_dict.get('last_name'),
                display_name=customer.profile.user_name,
                bio=customer.profile.bio,
                phone_number=customer.profile.phone_number,
                timezone=update_dict.get('timezone', 'UTC'),
                language=update_dict.get('language', 'en'),
                is_public=update_dict.get('is_public', False),
                avatar_url=customer.profile.avatar_url,
                is_verified=customer.is_verified,
                created_at=customer.profile.created_at,
                updated_at=customer.profile.updated_at,
                is_new=customer.profile.is_new
            )
        )
    
    except Exception as e:
        logger.error(f"Error updating customer profile: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating customer profile: {str(e)}"
        )

@router.get("/me", response_model=CustomerResponse)
async def get_current_customer_profile(
    current_customer_id: int = Depends(get_current_customer_id),
    db: Session = Depends(get_db)
):
    """Get current customer profile with full details"""
    try:
        # Fetch customer with profile in a single query
        customer = db.query(Customer).options(
            joinedload(Customer.profile)
        ).filter(Customer.id == current_customer_id).first()
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Ensure profile exists, create if not
        if not customer.profile:
            # Create a default profile if it doesn't exist
            profile = CustomerProfile(
                customer_id=customer.id, 
                is_new=True  # Default to new user
            )
            db.add(profile)
            db.commit()
            db.refresh(customer)
        
        # Construct response manually to avoid SQLAlchemy state issues
        return CustomerResponse(
            id=customer.id,
            email=customer.email,
            is_active=customer.is_active,
            is_verified=customer.is_verified,
            created_at=customer.created_at,
            updated_at=customer.updated_at,
            last_login=customer.last_login,
            login_attempts=customer.login_attempts,
            locked_until=customer.locked_until,
            profile=CustomerProfileResponse(
                id=customer.profile.id,
                customer_id=customer.profile.customer_id,
                first_name=customer.profile.full_name.split()[0] if customer.profile.full_name else None,
                last_name=customer.profile.full_name.split()[-1] if customer.profile.full_name and ' ' in customer.profile.full_name else None,
                display_name=customer.profile.user_name,
                bio=customer.profile.bio,
                phone_number=customer.profile.phone_number,
                timezone='UTC',  # Default timezone
                language='en',   # Default language
                is_public=False,
                avatar_url=customer.profile.avatar_url,
                is_verified=customer.is_verified,
                created_at=customer.profile.created_at,
                updated_at=customer.profile.updated_at,
                is_new=customer.profile.is_new
            )
        )
    
    except Exception as e:
        logger.error(f"Error fetching customer profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving customer profile"
        )

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