"""Example Integration of Enhanced Redis Manager with FastAPI

This module demonstrates how to properly integrate the Redis manager
with FastAPI services for graceful degradation and better error handling.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any, Optional
from .redis_manager import get_redis_manager, RedisManager
from .health_check import get_health_checker, HealthChecker
from .rate_limiting import brute_force_protection, rate_limit
from .csrf_protection import csrf_protection

logger = logging.getLogger(__name__)

# Example FastAPI app setup
app = FastAPI(title="Eindr Service with Enhanced Redis")

# Dependency injection for Redis manager
def get_redis() -> RedisManager:
    """Dependency to get Redis manager"""
    return get_redis_manager()

def get_health() -> HealthChecker:
    """Dependency to get health checker"""
    return get_health_checker()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting service with enhanced Redis manager")
    redis_manager = get_redis_manager()
    health_status = redis_manager.get_status()
    
    if health_status["available"]:
        logger.info("Redis connection established successfully")
    else:
        logger.warning("Redis not available - service will use local fallbacks")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down service")

# Health check endpoints
@app.get("/health")
async def health_check(health_checker: HealthChecker = Depends(get_health)):
    """Comprehensive health check endpoint"""
    return health_checker.get_comprehensive_health()

@app.get("/health/redis")
async def redis_health_check(health_checker: HealthChecker = Depends(get_health)):
    """Redis-specific health check"""
    return health_checker.check_redis_health()

@app.post("/admin/redis/reconnect")
async def force_redis_reconnect(health_checker: HealthChecker = Depends(get_health)):
    """Force Redis reconnection (admin endpoint)"""
    return health_checker.force_redis_reconnect()

# Example caching endpoints
@app.get("/cache/{key}")
@rate_limit("api.read")
async def get_cached_value(key: str, redis_manager: RedisManager = Depends(get_redis)):
    """Get value from cache with graceful fallback"""
    value = redis_manager.get(key)
    
    if value is None:
        # Simulate fetching from database or external service
        value = f"computed_value_for_{key}"
        
        # Try to cache the result
        cached = redis_manager.set(key, value, ex=300)  # 5 minutes
        
        return {
            "key": key,
            "value": value,
            "cached": cached,
            "source": "computed",
            "redis_available": redis_manager.is_available
        }
    
    return {
        "key": key,
        "value": value,
        "source": "cache",
        "redis_available": redis_manager.is_available
    }

@app.post("/cache/{key}")
@rate_limit("api.write")
async def set_cached_value(
    key: str, 
    value: str, 
    ttl: Optional[int] = 300,
    redis_manager: RedisManager = Depends(get_redis)
):
    """Set value in cache with graceful fallback"""
    success = redis_manager.set(key, value, ex=ttl)
    
    return {
        "key": key,
        "value": value,
        "ttl": ttl,
        "success": success,
        "redis_available": redis_manager.is_available
    }

@app.delete("/cache/{key}")
@rate_limit("api.delete")
async def delete_cached_value(key: str, redis_manager: RedisManager = Depends(get_redis)):
    """Delete value from cache"""
    deleted_count = redis_manager.delete(key)
    
    return {
        "key": key,
        "deleted": deleted_count > 0,
        "redis_available": redis_manager.is_available
    }

# Example session management
@app.post("/session/create")
@rate_limit("auth.login")
async def create_session(user_id: str, redis_manager: RedisManager = Depends(get_redis)):
    """Create user session with Redis fallback"""
    import uuid
    import json
    from datetime import datetime, timedelta
    
    session_id = str(uuid.uuid4())
    session_data = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat()
    }
    
    session_key = f"session:{session_id}"
    success = redis_manager.set(session_key, json.dumps(session_data), ex=86400)  # 24 hours
    
    if not success and not redis_manager.is_available:
        # In a real application, you might store in database as fallback
        logger.warning(f"Redis unavailable, session {session_id} not cached")
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "cached": success,
        "redis_available": redis_manager.is_available
    }

@app.get("/session/{session_id}")
@rate_limit("api.read")
async def get_session(session_id: str, redis_manager: RedisManager = Depends(get_redis)):
    """Get session data with fallback"""
    import json
    
    session_key = f"session:{session_id}"
    session_data = redis_manager.get(session_key)
    
    if session_data:
        try:
            return {
                "session_id": session_id,
                "data": json.loads(session_data),
                "source": "redis"
            }
        except json.JSONDecodeError:
            logger.error(f"Invalid session data for {session_id}")
    
    # In a real application, you might check database as fallback
    raise HTTPException(status_code=404, detail="Session not found")

# Example rate limiting with Redis
@app.get("/api/limited-endpoint")
@rate_limit("api.read")
async def limited_endpoint():
    """Example endpoint with rate limiting"""
    return {"message": "This endpoint is rate limited"}

# Example CSRF protection
@app.post("/api/protected-action")
async def protected_action():
    """Example endpoint with CSRF protection"""
    # CSRF protection would be applied via middleware or decorator
    return {"message": "Action completed successfully"}

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with Redis status"""
    redis_manager = get_redis_manager()
    
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "redis_available": redis_manager.is_available,
            "timestamp": str(datetime.utcnow())
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)