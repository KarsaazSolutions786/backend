from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.models import HistoryLog
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger
import uuid

router = APIRouter()

# Pydantic models
class HistoryLogCreate(BaseModel):
    content: Optional[str] = None
    interaction_type: Optional[str] = None

class HistoryLogResponse(BaseModel):
    id: str
    user_id: str
    content: Optional[str]
    interaction_type: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

@router.post("/", response_model=HistoryLogResponse)
async def create_history_log(
    log_data: HistoryLogCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Create a new history log entry."""
    try:
        user_id = current_user["uid"]
        
        history_log = HistoryLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=log_data.content,
            interaction_type=log_data.interaction_type
        )
        
        db.add(history_log)
        db.commit()
        db.refresh(history_log)
        
        logger.info(f"Created history log: {history_log.id} for user: {user_id}")
        return HistoryLogResponse(
            id=str(history_log.id),
            user_id=str(history_log.user_id),
            content=history_log.content,
            interaction_type=history_log.interaction_type,
            created_at=history_log.created_at
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Create history log error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[HistoryLogResponse])
async def get_history_logs(
    interaction_type: Optional[str] = None,
    days: Optional[int] = 30,
    limit: Optional[int] = 100,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get history logs for the current user."""
    try:
        user_id = current_user["uid"]
        query = db.query(HistoryLog).filter(HistoryLog.user_id == user_id)
        
        # Filter by interaction type if provided
        if interaction_type:
            query = query.filter(HistoryLog.interaction_type == interaction_type)
        
        # Filter by date range
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(HistoryLog.created_at >= cutoff_date)
        
        # Order by creation date and limit results
        logs = query.order_by(HistoryLog.created_at.desc()).limit(limit).all()
        
        return [
            HistoryLogResponse(
                id=str(log.id),
                user_id=str(log.user_id),
                content=log.content,
                interaction_type=log.interaction_type,
                created_at=log.created_at
            )
            for log in logs
        ]
        
    except Exception as e:
        logger.error(f"Get history logs error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{log_id}", response_model=HistoryLogResponse)
async def get_history_log(
    log_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get a specific history log."""
    try:
        user_id = current_user["uid"]
        log = db.query(HistoryLog).filter(
            HistoryLog.id == log_id,
            HistoryLog.user_id == user_id
        ).first()
        
        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="History log not found"
            )
        
        return HistoryLogResponse(
            id=str(log.id),
            user_id=str(log.user_id),
            content=log.content,
            interaction_type=log.interaction_type,
            created_at=log.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get history log error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{log_id}")
async def delete_history_log(
    log_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Delete a history log."""
    try:
        user_id = current_user["uid"]
        log = db.query(HistoryLog).filter(
            HistoryLog.id == log_id,
            HistoryLog.user_id == user_id
        ).first()
        
        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="History log not found"
            )
        
        db.delete(log)
        db.commit()
        
        logger.info(f"Deleted history log: {log_id}")
        return {"message": "History log deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete history log error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/")
async def clear_history_logs(
    interaction_type: Optional[str] = None,
    days: Optional[int] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Clear history logs based on filters."""
    try:
        user_id = current_user["uid"]
        query = db.query(HistoryLog).filter(HistoryLog.user_id == user_id)
        
        # Filter by interaction type if provided
        if interaction_type:
            query = query.filter(HistoryLog.interaction_type == interaction_type)
        
        # Filter by date range if provided
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(HistoryLog.created_at < cutoff_date)
        
        # Count logs to be deleted
        count = query.count()
        
        # Delete the logs
        query.delete(synchronize_session=False)
        db.commit()
        
        logger.info(f"Cleared {count} history logs for user: {user_id}")
        return {"message": f"Cleared {count} history logs"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Clear history logs error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/stats/summary")
async def get_history_stats(
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get history statistics for the user."""
    try:
        user_id = current_user["uid"]
        
        # Get total count
        total_logs = db.query(HistoryLog).filter(HistoryLog.user_id == user_id).count()
        
        # Get count by interaction type
        interaction_types = db.query(
            HistoryLog.interaction_type,
            db.func.count(HistoryLog.id).label('count')
        ).filter(
            HistoryLog.user_id == user_id
        ).group_by(HistoryLog.interaction_type).all()
        
        # Get recent activity (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_logs = db.query(HistoryLog).filter(
            HistoryLog.user_id == user_id,
            HistoryLog.created_at >= seven_days_ago
        ).count()
        
        # Format interaction types
        interaction_counts = {
            interaction_type: count for interaction_type, count in interaction_types
        }
        
        return {
            "total_logs": total_logs,
            "recent_logs_7_days": recent_logs,
            "interaction_types": interaction_counts,
            "summary_generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get history stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 