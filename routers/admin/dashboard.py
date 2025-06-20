"""
Admin Dashboard Router
=====================

Real-time dashboard endpoints providing comprehensive KPIs and metrics including:
- Live user and content statistics
- Performance monitoring
- System health indicators
- Growth analytics
- AI pipeline metrics
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from core.admin_security import (
    get_any_admin, get_analyst_or_higher, audit_action,
    AdminUser, AuditAction
)
from utils.kpi_cache import KpiQueries, get_cache_stats, warm_cache, invalidate_kpi_cache
from models.admin_models import AdminRole

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/dashboard",
    tags=["Admin - Dashboard"],
    responses={
        401: {"description": "Unauthorized - Invalid admin token"},
        403: {"description": "Forbidden - Insufficient permissions"},
        500: {"description": "Internal server error"}
    }
)

@router.get(
    "/kpis",
    response_model=Dict[str, Any],
    summary="Get Dashboard KPIs",
    description="Get comprehensive dashboard KPIs including user metrics, content stats, and performance data"
)
@audit_action(AuditAction.VIEW, target_type="dashboard", description="Viewed dashboard KPIs")
async def get_dashboard_kpis(
    request: Request,
    current_admin: AdminUser = Depends(get_any_admin)
) -> Dict[str, Any]:
    """
    Get real-time dashboard KPIs with caching for performance.
    
    Returns comprehensive metrics including:
    - User statistics (total, active, new)
    - Content metrics (reminders, notes, ledger entries)
    - AI pipeline performance
    - System health indicators
    """
    try:
        logger.info(f"Admin {current_admin.email} requested dashboard KPIs")
        
        # Get cached dashboard overview
        overview = await KpiQueries.get_dashboard_overview()
        
        # Get performance summary
        performance = await KpiQueries.get_performance_summary()
        
        # Get weekly growth metrics
        growth = await KpiQueries.get_weekly_growth_metrics()
        
        return {
            "success": True,
            "data": {
                "overview": overview,
                "performance": performance,
                "growth": growth,
                "last_updated": datetime.utcnow().isoformat(),
                "admin_viewer": {
                    "id": str(current_admin.id),
                    "name": current_admin.name,
                    "role": current_admin.role.value
                }
            },
            "cache_info": {
                "data_source": overview.get("data_source", "realtime"),
                "cache_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Dashboard KPIs request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve dashboard KPIs: {str(e)}"
        )

@router.get(
    "/usage-analytics",
    response_model=Dict[str, Any],
    summary="Get Usage Analytics",
    description="Get detailed usage analytics including hourly patterns, feature usage, and trends"
)
@audit_action(AuditAction.VIEW, target_type="analytics", description="Viewed usage analytics")
async def get_usage_analytics(
    request: Request,
    hours: int = Query(default=24, ge=1, le=168, description="Hours of data to retrieve (1-168)"),
    current_admin: AdminUser = Depends(get_analyst_or_higher)
) -> Dict[str, Any]:
    """
    Get detailed usage analytics for the specified time period.
    
    Args:
        hours: Number of hours of data to retrieve (1-168 hours = 1 week max)
        
    Returns:
        Detailed usage statistics and trends
    """
    try:
        logger.info(f"Admin {current_admin.email} requested usage analytics for {hours} hours")
        
        # Get hourly usage stats
        usage_stats = await KpiQueries.get_hourly_usage_stats(hours)
        
        # Calculate summary statistics
        total_requests = sum(stat.get("api_requests", 0) for stat in usage_stats)
        total_ai_requests = sum(stat.get("ai_pipeline_requests", 0) for stat in usage_stats)
        avg_response_time = sum(stat.get("avg_response_time_ms", 0) for stat in usage_stats) / len(usage_stats) if usage_stats else 0
        
        # Calculate trends (compare first half vs second half)
        if len(usage_stats) >= 4:
            mid_point = len(usage_stats) // 2
            recent_half = usage_stats[:mid_point]
            older_half = usage_stats[mid_point:]
            
            recent_avg_requests = sum(stat.get("api_requests", 0) for stat in recent_half) / len(recent_half)
            older_avg_requests = sum(stat.get("api_requests", 0) for stat in older_half) / len(older_half)
            
            trend_percentage = 0.0
            if older_avg_requests > 0:
                trend_percentage = ((recent_avg_requests - older_avg_requests) / older_avg_requests) * 100
        else:
            trend_percentage = 0.0
        
        return {
            "success": True,
            "data": {
                "time_period": {
                    "hours": hours,
                    "start_time": (datetime.utcnow() - timedelta(hours=hours)).isoformat(),
                    "end_time": datetime.utcnow().isoformat()
                },
                "summary": {
                    "total_api_requests": total_requests,
                    "total_ai_requests": total_ai_requests,
                    "average_response_time_ms": round(avg_response_time, 2),
                    "data_points": len(usage_stats),
                    "trend_percentage": round(trend_percentage, 2)
                },
                "hourly_data": usage_stats,
                "insights": {
                    "busiest_hour": max(usage_stats, key=lambda x: x.get("api_requests", 0)) if usage_stats else None,
                    "slowest_hour": max(usage_stats, key=lambda x: x.get("avg_response_time_ms", 0)) if usage_stats else None,
                    "ai_usage_percentage": round((total_ai_requests / total_requests * 100) if total_requests > 0 else 0, 2)
                }
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Usage analytics request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve usage analytics: {str(e)}"
        )

@router.get(
    "/system-health",
    response_model=Dict[str, Any],
    summary="Get System Health",
    description="Get real-time system health indicators and performance metrics"
)
@audit_action(AuditAction.VIEW, target_type="system_health", description="Viewed system health")
async def get_system_health(
    request: Request,
    current_admin: AdminUser = Depends(get_any_admin)
) -> Dict[str, Any]:
    """
    Get comprehensive system health indicators.
    
    Returns:
        Real-time system health metrics and status indicators
    """
    try:
        logger.info(f"Admin {current_admin.email} requested system health")
        
        # Get performance summary
        performance = await KpiQueries.get_performance_summary()
        
        # Get cache statistics
        cache_stats = await get_cache_stats()
        
        # Determine overall system status
        error_rate = performance.get("error_rate_percentage", 0)
        response_time = performance.get("avg_response_time_ms", 0)
        
        overall_status = "healthy"
        if error_rate > 5.0:
            overall_status = "degraded"
        elif error_rate > 10.0 or response_time > 2000:
            overall_status = "critical"
        
        status_indicators = []
        
        # API Performance
        if response_time < 500:
            status_indicators.append({"component": "API Response Time", "status": "healthy", "value": f"{response_time}ms"})
        elif response_time < 1000:
            status_indicators.append({"component": "API Response Time", "status": "warning", "value": f"{response_time}ms"})
        else:
            status_indicators.append({"component": "API Response Time", "status": "critical", "value": f"{response_time}ms"})
        
        # Error Rate
        if error_rate < 1.0:
            status_indicators.append({"component": "Error Rate", "status": "healthy", "value": f"{error_rate}%"})
        elif error_rate < 5.0:
            status_indicators.append({"component": "Error Rate", "status": "warning", "value": f"{error_rate}%"})
        else:
            status_indicators.append({"component": "Error Rate", "status": "critical", "value": f"{error_rate}%"})
        
        # Cache Health
        cache_hit_rate = cache_stats.get("hit_rate_percentage", 0)
        if cache_hit_rate > 80:
            status_indicators.append({"component": "Cache Hit Rate", "status": "healthy", "value": f"{cache_hit_rate}%"})
        elif cache_hit_rate > 60:
            status_indicators.append({"component": "Cache Hit Rate", "status": "warning", "value": f"{cache_hit_rate}%"})
        else:
            status_indicators.append({"component": "Cache Hit Rate", "status": "critical", "value": f"{cache_hit_rate}%"})
        
        return {
            "success": True,
            "data": {
                "overall_status": overall_status,
                "status_indicators": status_indicators,
                "performance_metrics": performance,
                "cache_health": cache_stats,
                "uptime_info": {
                    "status": "operational",
                    "last_restart": None,  # Would be tracked separately
                    "version": "1.0.0"  # Would come from app config
                },
                "health_score": min(100, max(0, 100 - (error_rate * 10) - (max(0, response_time - 200) / 20)))
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System health request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system health: {str(e)}"
        )

@router.get(
    "/growth-metrics",
    response_model=Dict[str, Any],
    summary="Get Growth Metrics",
    description="Get detailed growth analytics including user acquisition, retention, and engagement trends"
)
@audit_action(AuditAction.VIEW, target_type="growth_metrics", description="Viewed growth metrics")
async def get_growth_metrics(
    request: Request,
    period: str = Query(default="weekly", pattern="^(weekly|monthly)$", description="Time period for growth analysis"),
    current_admin: AdminUser = Depends(get_analyst_or_higher)
) -> Dict[str, Any]:
    """
    Get comprehensive growth metrics and trends.
    
    Args:
        period: Time period for analysis ("weekly" or "monthly")
        
    Returns:
        Detailed growth analytics and projections
    """
    try:
        logger.info(f"Admin {current_admin.email} requested growth metrics for {period} period")
        
        # Get weekly growth metrics (would extend for monthly)
        growth_data = await KpiQueries.get_weekly_growth_metrics()
        
        # Get current overview for context
        overview = await KpiQueries.get_dashboard_overview()
        
        # Calculate growth rates and projections
        current_users = growth_data.get("current_week_new_users", 0)
        previous_users = growth_data.get("previous_week_new_users", 0)
        growth_rate = growth_data.get("growth_rate_percentage", 0)
        
        # Project next period (simple linear projection)
        if period == "weekly":
            projected_next_period = current_users + (current_users * growth_rate / 100) if growth_rate > 0 else current_users
            period_label = "week"
        else:
            projected_next_period = current_users * 4.33 + (current_users * 4.33 * growth_rate / 100) if growth_rate > 0 else current_users * 4.33
            period_label = "month"
        
        return {
            "success": True,
            "data": {
                "period": period,
                "current_period": {
                    "new_users": current_users,
                    "period_label": f"Current {period_label}",
                    "total_users": overview.get("total_users", 0)
                },
                "previous_period": {
                    "new_users": previous_users,
                    "period_label": f"Previous {period_label}"
                },
                "growth_analysis": {
                    "growth_rate_percentage": growth_rate,
                    "trend": growth_data.get("trend", "flat"),
                    "projected_next_period": round(projected_next_period),
                    "is_accelerating": growth_rate > 0,
                    "growth_category": (
                        "rapid" if growth_rate > 20
                        else "moderate" if growth_rate > 5
                        else "slow" if growth_rate > 0
                        else "declining"
                    )
                },
                "retention_metrics": {
                    "retention_7d": overview.get("retention_rate_7d", 0),
                    "retention_30d": overview.get("retention_rate_30d", 0),
                    "retention_health": "good" if overview.get("retention_rate_7d", 0) > 40 else "needs_improvement"
                },
                "engagement_indicators": {
                    "active_users_today": overview.get("active_users_today", 0),
                    "content_creation_today": (
                        overview.get("reminders_created_today", 0) +
                        overview.get("notes_created_today", 0)
                    ),
                    "ai_usage_today": overview.get("ai_requests_today", 0)
                }
            },
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Growth metrics request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve growth metrics: {str(e)}"
        )

@router.post(
    "/cache/refresh",
    response_model=Dict[str, Any],
    summary="Refresh Dashboard Cache",
    description="Manually refresh dashboard cache for immediate data updates"
)
@audit_action(AuditAction.UPDATE, target_type="cache", description="Refreshed dashboard cache")
async def refresh_dashboard_cache(
    request: Request,
    pattern: Optional[str] = Query(default=None, description="Cache pattern to refresh (optional)"),
    current_admin: AdminUser = Depends(get_analyst_or_higher)
) -> Dict[str, Any]:
    """
    Manually refresh dashboard cache to get latest data.
    
    Args:
        pattern: Optional cache pattern to refresh (refreshes all if not specified)
        
    Returns:
        Cache refresh status and timing
    """
    try:
        logger.info(f"Admin {current_admin.email} requested cache refresh for pattern: {pattern}")
        
        start_time = datetime.utcnow()
        
        # Invalidate specified cache pattern
        await invalidate_kpi_cache(pattern)
        
        # Warm up critical caches
        await warm_cache()
        
        end_time = datetime.utcnow()
        refresh_duration = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "message": f"Cache refreshed successfully{f' for pattern: {pattern}' if pattern else ''}",
            "data": {
                "pattern_refreshed": pattern or "all",
                "refresh_duration_seconds": round(refresh_duration, 2),
                "refreshed_at": end_time.isoformat(),
                "refreshed_by": {
                    "admin_id": str(current_admin.id),
                    "admin_name": current_admin.name,
                    "admin_role": current_admin.role.value
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Cache refresh failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh cache: {str(e)}"
        )

@router.get(
    "/cache/stats",
    response_model=Dict[str, Any],
    summary="Get Cache Statistics",
    description="Get detailed cache performance statistics and health metrics"
)
@audit_action(AuditAction.VIEW, target_type="cache_stats", description="Viewed cache statistics")
async def get_cache_statistics(
    request: Request,
    current_admin: AdminUser = Depends(get_analyst_or_higher)
) -> Dict[str, Any]:
    """
    Get detailed cache performance statistics.
    
    Returns:
        Comprehensive cache metrics and performance data
    """
    try:
        logger.info(f"Admin {current_admin.email} requested cache statistics")
        
        cache_stats = await get_cache_stats()
        
        # Determine cache health
        hit_rate = cache_stats.get("hit_rate_percentage", 0)
        redis_status = cache_stats.get("redis_status", "unknown")
        
        cache_health = "excellent" if hit_rate > 90 else "good" if hit_rate > 70 else "needs_improvement"
        
        recommendations = []
        if hit_rate < 70:
            recommendations.append("Consider increasing cache TTL for frequently accessed data")
        if redis_status != "connected":
            recommendations.append("Redis connection issues detected - using memory fallback")
        if cache_stats.get("redis_errors", 0) > 10:
            recommendations.append("High Redis error rate - check Redis server health")
        
        return {
            "success": True,
            "data": {
                "performance": cache_stats,
                "health_assessment": {
                    "overall_health": cache_health,
                    "redis_operational": redis_status == "connected",
                    "recommendations": recommendations
                },
                "configuration": {
                    "cache_prefix": cache_stats.get("cache_prefix"),
                    "default_ttl_seconds": cache_stats.get("default_ttl_seconds"),
                    "redis_enabled": redis_status != "unavailable"
                }
            },
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache statistics request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )

@router.get(
    "/export",
    response_model=Dict[str, Any],
    summary="Export Dashboard Data",
    description="Export dashboard data in various formats for analysis"
)
@audit_action(AuditAction.EXPORT, target_type="dashboard_data", description="Exported dashboard data")
async def export_dashboard_data(
    request: Request,
    format: str = Query(default="json", pattern="^(json|csv)$", description="Export format"),
    include_hourly: bool = Query(default=False, description="Include hourly usage data"),
    hours: int = Query(default=24, ge=1, le=168, description="Hours of data to include"),
    current_admin: AdminUser = Depends(get_analyst_or_higher)
) -> Dict[str, Any]:
    """
    Export comprehensive dashboard data for external analysis.
    
    Args:
        format: Export format ("json" or "csv")
        include_hourly: Whether to include detailed hourly data
        hours: Hours of historical data to include
        
    Returns:
        Exported data in requested format
    """
    try:
        logger.info(f"Admin {current_admin.email} requested dashboard data export in {format} format")
        
        # Gather all dashboard data
        export_data = {
            "export_info": {
                "generated_at": datetime.utcnow().isoformat(),
                "generated_by": current_admin.name,
                "admin_role": current_admin.role.value,
                "format": format,
                "time_range_hours": hours
            }
        }
        
        # Get core dashboard data
        export_data["overview"] = await KpiQueries.get_dashboard_overview()
        export_data["performance"] = await KpiQueries.get_performance_summary()
        export_data["growth"] = await KpiQueries.get_weekly_growth_metrics()
        
        # Include hourly data if requested
        if include_hourly:
            export_data["hourly_usage"] = await KpiQueries.get_hourly_usage_stats(hours)
        
        # For CSV format, flatten the data structure
        if format == "csv":
            # This would be implemented to convert to CSV format
            # For now, return JSON with a note about CSV conversion
            export_data["note"] = "CSV export would be implemented to flatten this data structure"
        
        return {
            "success": True,
            "data": export_data,
            "export_summary": {
                "total_data_points": len(export_data.get("hourly_usage", [])) if include_hourly else 0,
                "includes_hourly_data": include_hourly,
                "export_size_estimate": "small"  # Would calculate actual size
            }
        }
        
    except Exception as e:
        logger.error(f"Dashboard data export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export dashboard data: {str(e)}"
        ) 