from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.models import Expense
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger

router = APIRouter()

class ExpenseCreate(BaseModel):
    amount: float
    description: str
    category: Optional[str] = "general"

class ExpenseUpdate(BaseModel):
    amount: Optional[float] = None
    description: Optional[str] = None
    category: Optional[str] = None

class ExpenseResponse(BaseModel):
    id: str
    amount: float
    description: str
    category: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

@router.post("/expenses", response_model=ExpenseResponse)
async def create_expense(
    expense_data: ExpenseCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Create a new expense."""
    try:
        expense = Expense(
            amount=expense_data.amount,
            description=expense_data.description,
            category=expense_data.category,
            user_id=current_user["uid"]
        )
        
        db.add(expense)
        db.commit()
        db.refresh(expense)
        
        logger.info(f"Created expense: {expense.id} for user: {current_user['uid']}")
        return expense
        
    except Exception as e:
        db.rollback()
        logger.error(f"Create expense error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/expenses", response_model=List[ExpenseResponse])
async def get_expenses(
    category: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all expenses for the current user."""
    try:
        query = db.query(Expense).filter(Expense.user_id == current_user["uid"])
        
        if category:
            query = query.filter(Expense.category == category)
        
        expenses = query.order_by(Expense.created_at.desc()).all()
        return expenses
        
    except Exception as e:
        logger.error(f"Get expenses error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/expenses/{expense_id}", response_model=ExpenseResponse)
async def get_expense(
    expense_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get a specific expense."""
    try:
        expense = db.query(Expense).filter(
            Expense.id == expense_id,
            Expense.user_id == current_user["uid"]
        ).first()
        
        if not expense:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expense not found"
            )
        
        return expense
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get expense error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/expenses/{expense_id}", response_model=ExpenseResponse)
async def update_expense(
    expense_id: str,
    expense_data: ExpenseUpdate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Update an expense."""
    try:
        expense = db.query(Expense).filter(
            Expense.id == expense_id,
            Expense.user_id == current_user["uid"]
        ).first()
        
        if not expense:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expense not found"
            )
        
        # Update fields if provided
        update_data = expense_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(expense, field, value)
        
        db.commit()
        db.refresh(expense)
        
        logger.info(f"Updated expense: {expense_id}")
        return expense
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update expense error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/expenses/{expense_id}")
async def delete_expense(
    expense_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Delete an expense."""
    try:
        expense = db.query(Expense).filter(
            Expense.id == expense_id,
            Expense.user_id == current_user["uid"]
        ).first()
        
        if not expense:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expense not found"
            )
        
        db.delete(expense)
        db.commit()
        
        logger.info(f"Deleted expense: {expense_id}")
        return {"message": "Expense deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete expense error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 