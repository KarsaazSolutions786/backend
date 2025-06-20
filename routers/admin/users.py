"""
Admin User Management Router
============================

Comprehensive user management system for admin panel including:
- User CRUD operations with role-based access
- Advanced search and filtering
- Bulk user operations
- User analytics and insights
- Account status management
- Data export capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, text, distinct
from sqlalchemy.orm import selectinload
import logging
import io
import csv
import json

from core.admin_security import (
    get_support_or_super, get_analyst_or_higher, get_super_admin,
    audit_action, AdminUser as AdminUserModel, AuditAction
)
from models.admin_models import AdminRole
from connect_db import get_db

# Import existing user models (adjust import path as needed)
# from models.user_models import User, UserProfile, UserStats
# For now, we'll create placeholder models - replace with actual imports

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/users",
    tags=["Admin - User Management"],
    responses={
        401: {"description": "Unauthorized - Invalid admin token"},
        403: {"description": "Forbidden - Insufficient permissions"},
        404: {"description": "User not found"},
        500: {"description": "Internal server error"}
    }
)

# Pydantic models for request/response
class UserSearchFilters(BaseModel):
    search_query: Optional[str] = Field(None, description="Search in name, email, or ID")
    email_domain: Optional[str] = Field(None, description="Filter by email domain")
    account_status: Optional[str] = Field(None, pattern="^(active|inactive|suspended|pending)$")
    subscription_status: Optional[str] = Field(None, pattern="^(trial|premium|free|expired)$")
    created_after: Optional[datetime] = Field(None, description="Filter users created after this date")
    created_before: Optional[datetime] = Field(None, description="Filter users created before this date")
    last_active_after: Optional[datetime] = Field(None, description="Filter by last activity")
    has_completed_trial: Optional[bool] = Field(None, description="Filter by trial completion status")
    min_content_count: Optional[int] = Field(None, ge=0, description="Minimum number of reminders/notes")
    
class BulkActionRequest(BaseModel):
    user_ids: List[str] = Field(..., min_length=1, max_length=100)
    action: str = Field(..., pattern="^(activate|deactivate|suspend|delete|send_notification)$")
    reason: Optional[str] = Field(None, description="Reason for bulk action")
    notification_data: Optional[Dict[str, Any]] = Field(None, description="Data for notification action")

class UserUpdateRequest(BaseModel):
    account_status: Optional[str] = Field(None, pattern="^(active|inactive|suspended)$")
    subscription_status: Optional[str] = Field(None, pattern="^(trial|premium|free|expired)$")
    notes: Optional[str] = Field(None, max_length=1000, description="Admin notes about user")
    force_password_reset: Optional[bool] = Field(None, description="Force user to reset password")

class UserStatsResponse(BaseModel):
    total_users: int
    active_users: int
    new_users_today: int
    new_users_this_week: int
    trial_users: int
    premium_users: int
    suspended_users: int
    avg_content_per_user: float
    retention_rate_7d: float
    retention_rate_30d: float

@router.get(
    "/stats",
    response_model=UserStatsResponse,
    summary="Get User Statistics",
    description="Get comprehensive user statistics and KPIs"
)
@audit_action(AuditAction.VIEW, target_type="user_stats", description="Viewed user statistics")
async def get_user_statistics(
    request: Request,
    current_admin: AdminUserModel = Depends(get_analyst_or_higher)
) -> UserStatsResponse:
    """
    Get comprehensive user statistics for admin dashboard.
    
    Returns:
        Detailed user metrics and KPIs
    """
    try:
        # async with get_db() as db:
        #     # This is a placeholder implementation
        #     # Replace with actual user model queries
        
        # For now, return mock data - replace with real queries
        stats = UserStatsResponse(
            total_users=1250,
            active_users=980,
            new_users_today=15,
            new_users_this_week=105,
            trial_users=320,
            premium_users=180,
            suspended_users=12,
            avg_content_per_user=8.5,
            retention_rate_7d=75.2,
            retention_rate_30d=45.8
        )
        
        logger.info(f"Admin {current_admin.email} viewed user statistics")
        return stats
            
    except Exception as e:
        logger.error(f"Failed to retrieve user statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user statistics: {str(e)}"
        )

@router.get(
    "/search",
    response_model=Dict[str, Any],
    summary="Search Users",
    description="Advanced user search with filtering and pagination"
)
@audit_action(AuditAction.VIEW, target_type="user_search", description="Searched users")
async def search_users(
    request: Request,
    filters: UserSearchFilters = Depends(),
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query(default="created_at", pattern="^(created_at|last_active|email|content_count)$"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
    current_admin: AdminUserModel = Depends(get_support_or_super)
) -> Dict[str, Any]:
    """
    Advanced user search with comprehensive filtering options.
    
    Args:
        filters: Search and filter criteria
        page: Page number for pagination
        limit: Number of results per page
        sort_by: Field to sort by
        sort_order: Sort order (asc/desc)
        
    Returns:
        Paginated user search results with metadata
    """
    try:
        # async with get_db() as db:
        #     # This is a placeholder implementation
        #     # Replace with actual user model queries when available
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Mock user data - replace with real database queries
        mock_users = []
        for i in range(1, min(limit + 1, 21)):
            user_id = f"user-{offset + i:04d}"
            mock_users.append({
                "id": user_id,
                "email": f"user{offset + i}@example.com",
                "name": f"User {offset + i}",
                "created_at": (datetime.utcnow() - timedelta(days=offset + i)).isoformat(),
                "last_active": (datetime.utcnow() - timedelta(hours=offset + i)).isoformat(),
                "account_status": "active" if i % 5 != 0 else "inactive",
                "subscription_status": "trial" if i % 3 == 0 else "free",
                "content_count": (i * 3) % 20,
                "trial_completed": i % 4 == 0,
                "admin_notes": f"Test user {offset + i}" if i % 7 == 0 else None
            })
        
        # Apply search filters (mock implementation)
        if filters.search_query:
            query_lower = filters.search_query.lower()
            mock_users = [
                user for user in mock_users
                if query_lower in user["email"].lower() or 
                   query_lower in user["name"].lower() or
                   query_lower in user["id"].lower()
            ]
        
        if filters.account_status:
            mock_users = [
                user for user in mock_users
                if user["account_status"] == filters.account_status
            ]
        
        if filters.subscription_status:
            mock_users = [
                user for user in mock_users
                if user["subscription_status"] == filters.subscription_status
            ]
        
        # Sort results
        reverse = sort_order == "desc"
        if sort_by == "created_at":
            mock_users.sort(key=lambda x: x["created_at"], reverse=reverse)
        elif sort_by == "email":
            mock_users.sort(key=lambda x: x["email"], reverse=reverse)
        elif sort_by == "content_count":
            mock_users.sort(key=lambda x: x["content_count"], reverse=reverse)
        
        # Calculate total (mock)
        total_count = 1250  # Would be actual count from database
        filtered_count = len(mock_users) if filters.search_query or filters.account_status else total_count
        
        logger.info(f"Admin {current_admin.email} searched users with filters")
        
        return {
            "success": True,
            "data": {
                "users": mock_users,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total_items": filtered_count,
                    "total_pages": (filtered_count + limit - 1) // limit,
                    "has_next": page * limit < filtered_count,
                    "has_previous": page > 1
                },
                "search_metadata": {
                    "filters_applied": {
                        "search_query": filters.search_query,
                        "account_status": filters.account_status,
                        "subscription_status": filters.subscription_status,
                        "date_range": {
                            "created_after": filters.created_after.isoformat() if filters.created_after else None,
                            "created_before": filters.created_before.isoformat() if filters.created_before else None
                        }
                    },
                    "sort": {
                        "field": sort_by,
                        "order": sort_order
                    },
                    "result_count": len(mock_users),
                    "total_database_users": total_count
                }
            }
        }
        
    except Exception as e:
        logger.error(f"User search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search users: {str(e)}"
        )

@router.get(
    "/{user_id}",
    response_model=Dict[str, Any],
    summary="Get User Details",
    description="Get comprehensive details for a specific user"
)
@audit_action(AuditAction.VIEW, target_type="user_details", description="Viewed user details")
async def get_user_details(
    user_id: str,
    request: Request,
    include_content: bool = Query(default=False, description="Include user's reminders/notes"),
    include_analytics: bool = Query(default=False, description="Include user analytics"),
    current_admin: AdminUserModel = Depends(get_support_or_super)
) -> Dict[str, Any]:
    """
    Get comprehensive details for a specific user.
    
    Args:
        user_id: User ID to retrieve
        include_content: Whether to include user's content
        include_analytics: Whether to include analytics data
        
    Returns:
        Detailed user information
    """
    try:
        # Mock user data - replace with actual database query
        mock_user = {
            "id": user_id,
            "email": f"user.{user_id}@example.com",
            "name": f"User {user_id}",
            "created_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "last_active": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "account_status": "active",
            "subscription_status": "trial",
            "trial_started": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "trial_expires": (datetime.utcnow() + timedelta(days=9)).isoformat(),
            "profile": {
                "phone": "+1234567890",
                "timezone": "America/New_York",
                "language": "en",
                "notification_preferences": {
                    "email": True,
                    "push": True,
                    "sms": False
                }
            },
            "admin_notes": "Power user, very engaged",
            "flags": {
                "email_verified": True,
                "phone_verified": False,
                "terms_accepted": True,
                "marketing_opted_in": True
            },
            "content_summary": {
                "total_reminders": 25,
                "total_notes": 12,
                "total_ledger_entries": 8,
                "ai_interactions": 45
            } if include_content else None,
            "analytics": {
                "session_count": 156,
                "avg_session_duration_minutes": 8.5,
                "total_time_spent_minutes": 1326,
                "features_used": ["reminders", "notes", "ai_assistant", "ledger"],
                "most_active_time": "evening",
                "device_info": {
                    "platform": "iOS",
                    "app_version": "1.2.3",
                    "last_device": "iPhone 14"
                }
            } if include_analytics else None
        }
        
        if not mock_user:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        # Log audit action with user ID
        request.state.audit_target_id = user_id
        
        logger.info(f"Admin {current_admin.email} viewed details for user {user_id}")
        
        return {
            "success": True,
            "data": mock_user,
            "permissions": {
                "can_edit": current_admin.role in [AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT],
                "can_delete": current_admin.role == AdminRole.SUPER_ADMIN,
                "can_suspend": current_admin.role in [AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT],
                "can_view_content": include_content,
                "can_view_analytics": include_analytics
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve user details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user details: {str(e)}"
        )

@router.put(
    "/{user_id}",
    response_model=Dict[str, Any],
    summary="Update User",
    description="Update user account status, subscription, and admin notes"
)
@audit_action(AuditAction.UPDATE, target_type="user", description="Updated user account")
async def update_user(
    user_id: str,
    update_data: UserUpdateRequest,
    request: Request,
    current_admin: AdminUserModel = Depends(get_support_or_super)
) -> Dict[str, Any]:
    """
    Update user account status and admin-controlled fields.
    
    Args:
        user_id: User ID to update
        update_data: Fields to update
        
    Returns:
        Updated user information
    """
    try:
        # Log audit action with user ID
        request.state.audit_target_id = user_id
        
        # Mock update - replace with actual database update
        changes_made = []
        
        if update_data.account_status:
            changes_made.append(f"Account status changed to {update_data.account_status}")
        
        if update_data.subscription_status:
            changes_made.append(f"Subscription status changed to {update_data.subscription_status}")
        
        if update_data.notes:
            changes_made.append("Admin notes updated")
        
        if update_data.force_password_reset:
            changes_made.append("Password reset forced")
            # Would trigger password reset email here
        
        if not changes_made:
            raise HTTPException(
                status_code=400,
                detail="No valid updates provided"
            )
        
        # Log detailed audit information
        request.state.audit_data = {
            "changes": changes_made,
            "update_data": update_data.dict(exclude_unset=True)
        }
        
        logger.info(f"Admin {current_admin.email} updated user {user_id}: {', '.join(changes_made)}")
        
        return {
            "success": True,
            "message": "User updated successfully",
            "data": {
                "user_id": user_id,
                "changes_made": changes_made,
                "updated_at": datetime.utcnow().isoformat(),
                "updated_by": {
                    "admin_id": str(current_admin.id),
                    "admin_name": current_admin.name
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update user: {str(e)}"
        )

@router.post(
    "/{user_id}/suspend",
    response_model=Dict[str, Any],
    summary="Suspend User",
    description="Suspend user account with reason"
)
@audit_action(AuditAction.UPDATE, target_type="user_suspension", description="Suspended user account")
async def suspend_user(
    user_id: str,
    request: Request,
    reason: str = Body(..., description="Reason for suspension"),
    duration_days: Optional[int] = Body(None, ge=1, le=365, description="Suspension duration in days"),
    current_admin: AdminUserModel = Depends(get_support_or_super)
) -> Dict[str, Any]:
    """
    Suspend a user account with specified reason and optional duration.
    """
    try:
        request.state.audit_target_id = user_id
        request.state.audit_data = {
            "reason": reason,
            "duration_days": duration_days,
            "suspended_by": current_admin.name
        }
        
        # Mock suspension - replace with actual implementation
        suspension_until = None
        if duration_days:
            suspension_until = datetime.utcnow() + timedelta(days=duration_days)
        
        logger.warning(f"Admin {current_admin.email} suspended user {user_id} - Reason: {reason}")
        
        return {
            "success": True,
            "message": "User suspended successfully",
            "data": {
                "user_id": user_id,
                "suspended_at": datetime.utcnow().isoformat(),
                "suspended_until": suspension_until.isoformat() if suspension_until else None,
                "reason": reason,
                "suspended_by": {
                    "admin_id": str(current_admin.id),
                    "admin_name": current_admin.name
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to suspend user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to suspend user: {str(e)}"
        )

@router.post(
    "/{user_id}/reactivate",
    response_model=Dict[str, Any],
    summary="Reactivate User",
    description="Reactivate suspended user account"
)
@audit_action(AuditAction.UPDATE, target_type="user_reactivation", description="Reactivated user account")
async def reactivate_user(
    user_id: str,
    request: Request,
    reason: str = Body(..., description="Reason for reactivation"),
    current_admin: AdminUserModel = Depends(get_support_or_super)
) -> Dict[str, Any]:
    """
    Reactivate a suspended user account.
    """
    try:
        request.state.audit_target_id = user_id
        request.state.audit_data = {
            "reason": reason,
            "reactivated_by": current_admin.name
        }
        
        # Mock reactivation - replace with actual implementation
        
        logger.info(f"Admin {current_admin.email} reactivated user {user_id} - Reason: {reason}")
        
        return {
            "success": True,
            "message": "User reactivated successfully",
            "data": {
                "user_id": user_id,
                "reactivated_at": datetime.utcnow().isoformat(),
                "reason": reason,
                "reactivated_by": {
                    "admin_id": str(current_admin.id),
                    "admin_name": current_admin.name
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to reactivate user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reactivate user: {str(e)}"
        )

@router.delete(
    "/{user_id}",
    response_model=Dict[str, Any],
    summary="Delete User",
    description="Permanently delete user account (Super Admin only)"
)
@audit_action(AuditAction.DELETE, target_type="user", description="Deleted user account")
async def delete_user(
    user_id: str,
    request: Request,
    reason: str = Body(..., description="Reason for deletion"),
    confirm_deletion: bool = Body(..., description="Confirmation that this is intentional"),
    current_admin: AdminUserModel = Depends(get_super_admin)
) -> Dict[str, Any]:
    """
    Permanently delete a user account. This action cannot be undone.
    Only Super Admins can perform this action.
    """
    try:
        if not confirm_deletion:
            raise HTTPException(
                status_code=400,
                detail="Deletion confirmation required"
            )
        
        request.state.audit_target_id = user_id
        request.state.audit_data = {
            "reason": reason,
            "deleted_by": current_admin.name,
            "confirmation": confirm_deletion
        }
        
        # Mock deletion - replace with actual implementation
        # This would:
        # 1. Anonymize user data according to GDPR
        # 2. Delete user content or transfer to system
        # 3. Remove from Firebase Auth
        # 4. Log detailed audit trail
        
        logger.critical(f"Admin {current_admin.email} deleted user {user_id} - Reason: {reason}")
        
        return {
            "success": True,
            "message": "User deleted successfully",
            "data": {
                "user_id": user_id,
                "deleted_at": datetime.utcnow().isoformat(),
                "reason": reason,
                "deleted_by": {
                    "admin_id": str(current_admin.id),
                    "admin_name": current_admin.name
                },
                "warning": "This action is permanent and cannot be undone"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete user: {str(e)}"
        )

@router.post(
    "/bulk-actions",
    response_model=Dict[str, Any],
    summary="Bulk User Actions",
    description="Perform bulk actions on multiple users"
)
@audit_action(AuditAction.UPDATE, target_type="users", description="Performed bulk user action")
async def bulk_user_actions(
    action_request: BulkActionRequest,
    request: Request,
    current_admin: AdminUserModel = Depends(get_support_or_super)
) -> Dict[str, Any]:
    """
    Perform bulk actions on multiple users.
    
    Supported actions:
    - activate: Activate user accounts
    - deactivate: Deactivate user accounts  
    - suspend: Suspend user accounts
    - delete: Delete user accounts (Super Admin only)
    - send_notification: Send notification to users
    """
    try:
        # Check permissions for delete action
        if action_request.action == "delete" and current_admin.role != AdminRole.SUPER_ADMIN:
            raise HTTPException(
                status_code=403,
                detail="Only Super Admins can perform bulk delete operations"
            )
        
        if len(action_request.user_ids) > 100:
            raise HTTPException(
                status_code=400,
                detail="Bulk actions limited to 100 users at a time"
            )
        
        # Log audit action
        request.state.audit_data = {
            "action": action_request.action,
            "user_count": len(action_request.user_ids),
            "user_ids": action_request.user_ids,
            "reason": action_request.reason,
            "performed_by": current_admin.name
        }
        
        # Mock bulk action processing
        successful_actions = []
        failed_actions = []
        
        for user_id in action_request.user_ids:
            try:
                # Mock processing for each user
                # Replace with actual implementation
                successful_actions.append({
                    "user_id": user_id,
                    "status": "success",
                    "message": f"Action '{action_request.action}' completed successfully"
                })
            except Exception as e:
                failed_actions.append({
                    "user_id": user_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Log results
        success_count = len(successful_actions)
        failure_count = len(failed_actions)
        
        logger.info(
            f"Admin {current_admin.email} performed bulk action '{action_request.action}' on {len(action_request.user_ids)} users - "
            f"Success: {success_count}, Failed: {failure_count}"
        )
        
        return {
            "success": True,
            "message": f"Bulk action '{action_request.action}' completed",
            "data": {
                "action": action_request.action,
                "total_users": len(action_request.user_ids),
                "successful_count": success_count,
                "failed_count": failure_count,
                "results": {
                    "successful": successful_actions,
                    "failed": failed_actions
                },
                "performed_at": datetime.utcnow().isoformat(),
                "performed_by": {
                    "admin_id": str(current_admin.id),
                    "admin_name": current_admin.name
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk user action failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform bulk action: {str(e)}"
        )

@router.get(
    "/export",
    summary="Export User Data",
    description="Export user data in CSV or JSON format"
)
@audit_action(AuditAction.EXPORT, target_type="user_data", description="Exported user data")
async def export_user_data(
    request: Request,
    format: str = Query(default="csv", pattern="^(csv|json)$", description="Export format"),
    filters: UserSearchFilters = Depends(),
    current_admin: AdminUserModel = Depends(get_analyst_or_higher)
) -> StreamingResponse:
    """
    Export user data in CSV or JSON format with optional filtering.
    """
    try:
        # Mock data export - replace with actual implementation
        mock_users = [
            {
                "id": f"user-{i:04d}",
                "email": f"user{i}@example.com",
                "name": f"User {i}",
                "created_at": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "account_status": "active" if i % 5 != 0 else "inactive",
                "subscription_status": "trial" if i % 3 == 0 else "free",
                "content_count": (i * 3) % 20,
                "last_active": (datetime.utcnow() - timedelta(hours=i)).isoformat()
            }
            for i in range(1, 101)  # Export first 100 users for demo
        ]
        
        # Apply filters (simplified for demo)
        if filters.account_status:
            mock_users = [u for u in mock_users if u["account_status"] == filters.account_status]
        
        logger.info(f"Admin {current_admin.email} exported {len(mock_users)} users in {format} format")
        
        if format == "csv":
            # Generate CSV
            output = io.StringIO()
            if mock_users:
                writer = csv.DictWriter(output, fieldnames=mock_users[0].keys())
                writer.writeheader()
                writer.writerows(mock_users)
            
            # Create streaming response
            def iter_csv():
                output.seek(0)
                yield output.getvalue()
            
            return StreamingResponse(
                iter_csv(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=users_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        
        else:  # JSON format
            export_data = {
                "export_info": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "generated_by": current_admin.name,
                    "format": format,
                    "total_users": len(mock_users)
                },
                "users": mock_users
            }
            
            def iter_json():
                yield json.dumps(export_data, indent=2)
            
            return StreamingResponse(
                iter_json(),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=users_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"}
            )
        
    except Exception as e:
        logger.error(f"User data export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export user data: {str(e)}"
        ) 