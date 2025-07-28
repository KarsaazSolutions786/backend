# Use secure shared authentication 
import sys
import os

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
                options={"verify_signature": True, "verify_exp": True}
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
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

class ActivityLog(BaseModel):
    id: str
    customer_id: str
    action: str
    resource_type: str  # 'reminder', 'note', 'expense', 'customer'
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime

class ActivityCreate(BaseModel):
    action: str = Field(..., min_length=1, max_length=100)
    resource_type: str = Field(..., min_length=1, max_length=50)
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = {}

class ActivityStats(BaseModel):
    total_activities: int
    actions_today: int
    most_common_actions: List[Dict[str, Any]]
    activity_by_hour: List[Dict[str, Any]]

# Mock storage
activity_logs_storage = {}

@router.post("/", response_model=ActivityLog)
async def log_activity(activity_data: ActivityCreate, request: Request = None):
    """Log a customer activity"""
    try:
        activity_id = str(uuid.uuid4())
        customer_id: int = Depends(get_current_customer_id)
        
        activity = {
            "id": activity_id,
            "customer_id": customer_id,
            "action": activity_data.action,
            "resource_type": activity_data.resource_type,
            "resource_id": activity_data.resource_id,
            "details": activity_data.details or {},
            "ip_address": "127.0.0.1",  # Would extract from request
            "user_agent": "MockAgent",    # Would extract from request
            "timestamp": datetime.utcnow()
        }
        
        activity_logs_storage[activity_id] = activity
        
        logger.info(f"Logged activity: {activity_data.action} for user: {customer_id}")
        
        return ActivityLog(**activity)
        
    except Exception as e:
        logger.error(f"Error logging activity: {e}")
        raise HTTPException(status_code=500, detail="Failed to log activity")

@router.get("/", response_model=List[ActivityLog])
async def get_activity_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None)
):
    """Get customer's activity logs"""
    try:
        customer_id: int = Depends(get_current_customer_id)
        user_activities = [activity for activity in activity_logs_storage.values() 
                          if activity["customer_id"] == customer_id]
        
        # Apply filters
        if action:
            user_activities = [a for a in user_activities if a["action"] == action]
        
        if resource_type:
            user_activities = [a for a in user_activities if a["resource_type"] == resource_type]
        
        if from_date:
            user_activities = [a for a in user_activities if a["timestamp"] >= from_date]
        
        if to_date:
            user_activities = [a for a in user_activities if a["timestamp"] <= to_date]
        
        # Sort by timestamp (newest first)
        user_activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Apply pagination
        paginated_activities = user_activities[skip:skip + limit]
        
        return [ActivityLog(**activity) for activity in paginated_activities]
        
    except Exception as e:
        logger.error(f"Error getting activity logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get activity logs")

@router.get("/stats", response_model=ActivityStats)
async def get_activity_stats():
    """Get activity statistics"""
    try:
        customer_id: int = Depends(get_current_customer_id)
        user_activities = [activity for activity in activity_logs_storage.values() 
                          if activity["customer_id"] == customer_id]
        
        total_activities = len(user_activities)
        
        # Activities today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        actions_today = len([a for a in user_activities if a["timestamp"] >= today_start])
        
        # Most common actions
        action_counts = {}
        for activity in user_activities:
            action = activity["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        most_common_actions = [
            {"action": action, "count": count}
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Activity by hour (last 24 hours)
        activity_by_hour = []
        now = datetime.utcnow()
        for i in range(24):
            hour_start = now - timedelta(hours=i+1)
            hour_end = now - timedelta(hours=i)
            hour_activities = [a for a in user_activities 
                             if hour_start <= a["timestamp"] < hour_end]
            
            activity_by_hour.append({
                "hour": hour_start.strftime("%H:00"),
                "count": len(hour_activities)
            })
        
        return ActivityStats(
            total_activities=total_activities,
            actions_today=actions_today,
            most_common_actions=most_common_actions,
            activity_by_hour=activity_by_hour
        )
        
    except Exception as e:
        logger.error(f"Error getting activity stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get activity stats")

@router.delete("/cleanup")
async def cleanup_old_logs(days: int = Query(30, ge=1, le=365)):
    """Clean up old activity logs"""
    try:
        customer_id: int = Depends(get_current_customer_id)
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Remove old logs
        logs_to_remove = [
            log_id for log_id, log in activity_logs_storage.items()
            if log["customer_id"] == customer_id and log["timestamp"] < cutoff_date
        ]
        
        for log_id in logs_to_remove:
            del activity_logs_storage[log_id]
        
        return {"message": f"Cleaned up {len(logs_to_remove)} old activity logs"}
        
    except Exception as e:
        logger.error(f"Error cleaning up logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup logs")
