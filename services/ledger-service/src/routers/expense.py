from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models
class ExpenseCreate(BaseModel):
    amount: float = Field(..., gt=0)
    description: str = Field(..., min_length=1, max_length=500)
    category: str = Field(..., min_length=1, max_length=100)
    date: Optional[datetime] = None
    payment_method: Optional[str] = "cash"
    tags: Optional[List[str]] = []
    location: Optional[str] = None
    receipt_url: Optional[str] = None
    is_recurring: bool = False
    recurring_frequency: Optional[str] = None  # 'daily', 'weekly', 'monthly'

class ExpenseUpdate(BaseModel):
    amount: Optional[float] = Field(None, gt=0)
    description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=100)
    date: Optional[datetime] = None
    payment_method: Optional[str] = None
    tags: Optional[List[str]] = None
    location: Optional[str] = None
    receipt_url: Optional[str] = None
    is_recurring: Optional[bool] = None
    recurring_frequency: Optional[str] = None

class ExpenseResponse(BaseModel):
    id: str
    user_id: str
    amount: float
    description: str
    category: str
    date: datetime
    payment_method: str
    tags: List[str]
    location: Optional[str]
    receipt_url: Optional[str]
    is_recurring: bool
    recurring_frequency: Optional[str]
    created_at: datetime
    updated_at: datetime

class BudgetCreate(BaseModel):
    category: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0)
    period: str = Field(..., regex="^(daily|weekly|monthly|yearly)$")
    start_date: Optional[datetime] = None

class BudgetResponse(BaseModel):
    id: str
    category: str
    amount: float
    period: str
    spent_amount: float
    remaining_amount: float
    percentage_used: float
    start_date: datetime
    end_date: datetime
    is_exceeded: bool

class ExpenseStats(BaseModel):
    total_expenses: float
    expense_count: int
    average_expense: float
    top_categories: List[Dict[str, Any]]
    monthly_trend: List[Dict[str, Any]]
    payment_method_breakdown: Dict[str, float]

# Mock data storage
expenses_storage = {}
budgets_storage = {}

def get_current_user_id() -> str:
    """Mock function to get current user ID"""
    return "user-123"

@router.post("/", response_model=ExpenseResponse, status_code=status.HTTP_201_CREATED)
async def create_expense(expense_data: ExpenseCreate):
    """Create a new expense"""
    try:
        expense_id = str(uuid.uuid4())
        user_id = get_current_user_id()
        
        expense = {
            "id": expense_id,
            "user_id": user_id,
            "amount": expense_data.amount,
            "description": expense_data.description,
            "category": expense_data.category,
            "date": expense_data.date or datetime.utcnow(),
            "payment_method": expense_data.payment_method or "cash",
            "tags": expense_data.tags or [],
            "location": expense_data.location,
            "receipt_url": expense_data.receipt_url,
            "is_recurring": expense_data.is_recurring,
            "recurring_frequency": expense_data.recurring_frequency,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        expenses_storage[expense_id] = expense
        
        logger.info(f"Created expense: {expense_id} for user: {user_id} - ${expense_data.amount}")
        
        return ExpenseResponse(**expense)
        
    except Exception as e:
        logger.error(f"Error creating expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create expense"
        )

@router.get("/", response_model=List[ExpenseResponse])
async def get_expenses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = Query(None),
    payment_method: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    min_amount: Optional[float] = Query(None, ge=0),
    max_amount: Optional[float] = Query(None, ge=0),
    tags: Optional[str] = Query(None)  # Comma-separated tags
):
    """Get user's expenses with filtering options"""
    try:
        user_id = get_current_user_id()
        user_expenses = [expense for expense in expenses_storage.values() 
                        if expense["user_id"] == user_id]
        
        # Apply filters
        if category:
            user_expenses = [expense for expense in user_expenses 
                           if expense["category"].lower() == category.lower()]
        
        if payment_method:
            user_expenses = [expense for expense in user_expenses 
                           if expense["payment_method"].lower() == payment_method.lower()]
        
        if from_date:
            user_expenses = [expense for expense in user_expenses 
                           if expense["date"] >= from_date]
        
        if to_date:
            user_expenses = [expense for expense in user_expenses 
                           if expense["date"] <= to_date]
        
        if min_amount is not None:
            user_expenses = [expense for expense in user_expenses 
                           if expense["amount"] >= min_amount]
        
        if max_amount is not None:
            user_expenses = [expense for expense in user_expenses 
                           if expense["amount"] <= max_amount]
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            user_expenses = [expense for expense in user_expenses 
                           if any(tag in expense.get("tags", []) for tag in tag_list)]
        
        # Sort by date (newest first)
        user_expenses.sort(key=lambda x: x["date"], reverse=True)
        
        # Apply pagination
        paginated_expenses = user_expenses[skip:skip + limit]
        
        return [ExpenseResponse(**expense) for expense in paginated_expenses]
        
    except Exception as e:
        logger.error(f"Error getting expenses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get expenses"
        )

@router.get("/{expense_id}", response_model=ExpenseResponse)
async def get_expense(expense_id: str):
    """Get a specific expense"""
    try:
        expense = expenses_storage.get(expense_id)
        
        if not expense:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expense not found"
            )
        
        # Check ownership
        user_id = get_current_user_id()
        if expense["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return ExpenseResponse(**expense)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get expense"
        )

@router.put("/{expense_id}", response_model=ExpenseResponse)
async def update_expense(expense_id: str, expense_data: ExpenseUpdate):
    """Update an expense"""
    try:
        expense = expenses_storage.get(expense_id)
        
        if not expense:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expense not found"
            )
        
        # Check ownership
        user_id = get_current_user_id()
        if expense["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Update fields
        update_data = expense_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            expense[field] = value
        
        expense["updated_at"] = datetime.utcnow()
        
        logger.info(f"Updated expense: {expense_id}")
        
        return ExpenseResponse(**expense)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update expense"
        )

@router.delete("/{expense_id}")
async def delete_expense(expense_id: str):
    """Delete an expense"""
    try:
        expense = expenses_storage.get(expense_id)
        
        if not expense:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expense not found"
            )
        
        # Check ownership
        user_id = get_current_user_id()
        if expense["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        del expenses_storage[expense_id]
        
        logger.info(f"Deleted expense: {expense_id}")
        
        return {"message": "Expense deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete expense"
        )

@router.get("/stats/summary", response_model=ExpenseStats)
async def get_expense_stats():
    """Get expense statistics and analytics"""
    try:
        user_id = get_current_user_id()
        user_expenses = [expense for expense in expenses_storage.values() 
                        if expense["user_id"] == user_id]
        
        if not user_expenses:
            return ExpenseStats(
                total_expenses=0,
                expense_count=0,
                average_expense=0,
                top_categories=[],
                monthly_trend=[],
                payment_method_breakdown={}
            )
        
        # Calculate basic stats
        total_expenses = sum(expense["amount"] for expense in user_expenses)
        expense_count = len(user_expenses)
        average_expense = total_expenses / expense_count if expense_count > 0 else 0
        
        # Top categories
        category_totals = {}
        for expense in user_expenses:
            category = expense["category"]
            category_totals[category] = category_totals.get(category, 0) + expense["amount"]
        
        top_categories = [
            {"category": cat, "amount": amount, "count": sum(1 for e in user_expenses if e["category"] == cat)}
            for cat, amount in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Monthly trend (last 6 months)
        monthly_trend = []
        for i in range(6):
            month_start = datetime.utcnow().replace(day=1) - timedelta(days=30 * i)
            month_end = month_start + timedelta(days=30)
            
            month_expenses = [e for e in user_expenses if month_start <= e["date"] < month_end]
            month_total = sum(e["amount"] for e in month_expenses)
            
            monthly_trend.append({
                "month": month_start.strftime("%Y-%m"),
                "amount": month_total,
                "count": len(month_expenses)
            })
        
        # Payment method breakdown
        payment_method_breakdown = {}
        for expense in user_expenses:
            method = expense["payment_method"]
            payment_method_breakdown[method] = payment_method_breakdown.get(method, 0) + expense["amount"]
        
        return ExpenseStats(
            total_expenses=total_expenses,
            expense_count=expense_count,
            average_expense=average_expense,
            top_categories=top_categories,
            monthly_trend=monthly_trend,
            payment_method_breakdown=payment_method_breakdown
        )
        
    except Exception as e:
        logger.error(f"Error getting expense stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get expense statistics"
        )

@router.post("/budgets", response_model=BudgetResponse)
async def create_budget(budget_data: BudgetCreate):
    """Create a budget for a category"""
    try:
        budget_id = str(uuid.uuid4())
        user_id = get_current_user_id()
        
        start_date = budget_data.start_date or datetime.utcnow()
        
        # Calculate end date based on period
        if budget_data.period == "daily":
            end_date = start_date + timedelta(days=1)
        elif budget_data.period == "weekly":
            end_date = start_date + timedelta(weeks=1)
        elif budget_data.period == "monthly":
            end_date = start_date + timedelta(days=30)
        else:  # yearly
            end_date = start_date + timedelta(days=365)
        
        budget = {
            "id": budget_id,
            "user_id": user_id,
            "category": budget_data.category,
            "amount": budget_data.amount,
            "period": budget_data.period,
            "start_date": start_date,
            "end_date": end_date,
            "created_at": datetime.utcnow()
        }
        
        budgets_storage[budget_id] = budget
        
        # Calculate current spending in this category for this period
        user_expenses = [expense for expense in expenses_storage.values() 
                        if expense["user_id"] == user_id and 
                           expense["category"] == budget_data.category and
                           start_date <= expense["date"] <= end_date]
        
        spent_amount = sum(expense["amount"] for expense in user_expenses)
        remaining_amount = budget_data.amount - spent_amount
        percentage_used = (spent_amount / budget_data.amount * 100) if budget_data.amount > 0 else 0
        is_exceeded = spent_amount > budget_data.amount
        
        return BudgetResponse(
            **budget,
            spent_amount=spent_amount,
            remaining_amount=remaining_amount,
            percentage_used=percentage_used,
            is_exceeded=is_exceeded
        )
        
    except Exception as e:
        logger.error(f"Error creating budget: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create budget"
        )

@router.get("/budgets", response_model=List[BudgetResponse])
async def get_budgets():
    """Get user's budgets with current spending"""
    try:
        user_id = get_current_user_id()
        user_budgets = [budget for budget in budgets_storage.values() 
                       if budget["user_id"] == user_id]
        
        result = []
        for budget in user_budgets:
            # Calculate current spending for this budget
            user_expenses = [expense for expense in expenses_storage.values() 
                           if expense["user_id"] == user_id and 
                              expense["category"] == budget["category"] and
                              budget["start_date"] <= expense["date"] <= budget["end_date"]]
            
            spent_amount = sum(expense["amount"] for expense in user_expenses)
            remaining_amount = budget["amount"] - spent_amount
            percentage_used = (spent_amount / budget["amount"] * 100) if budget["amount"] > 0 else 0
            is_exceeded = spent_amount > budget["amount"]
            
            result.append(BudgetResponse(
                **budget,
                spent_amount=spent_amount,
                remaining_amount=remaining_amount,
                percentage_used=percentage_used,
                is_exceeded=is_exceeded
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting budgets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get budgets"
        )

@router.get("/categories")
async def get_expense_categories():
    """Get all expense categories used by the user"""
    try:
        user_id = get_current_user_id()
        user_expenses = [expense for expense in expenses_storage.values() 
                        if expense["user_id"] == user_id]
        
        categories = list(set(expense["category"] for expense in user_expenses))
        categories.sort()
        
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get categories"
        )
