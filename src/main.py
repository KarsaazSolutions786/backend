"""
Main application entry point for Railway deployment
This serves as a unified API gateway for all microservices
"""

import os
import sys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import time
import jwt
from typing import Optional
from contextlib import asynccontextmanager

# Import microservice client
from .services.microservice_client import microservice_client

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))

# Import enhanced Redis manager and security modules
try:
    from redis_manager import get_redis_manager, RedisManager
    from health_check import get_health_checker, HealthChecker
    from rate_limiting import RateLimitMiddleware, rate_limit
    from csrf_protection import csrf_protection
    SECURITY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Security modules not available: {e}")
    SECURITY_AVAILABLE = False
    # Define dummy classes for type hints when imports fail
    class RedisManager:
        pass
    class HealthChecker:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for ML libraries availability
ML_AVAILABLE = False
try:
    import torch
    import transformers
    ML_AVAILABLE = True
    logger.info("ML libraries available")
except ImportError:
    logger.warning("ML libraries not available - running in API-only mode")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Eindr Microservices API...")
    
    if SECURITY_AVAILABLE:
        # Initialize Redis manager
        redis_manager = get_redis_manager()
        health_status = redis_manager.get_status()
        
        if health_status["available"]:
            logger.info("Redis connection established successfully")
        else:
            logger.warning("Redis not available - service will use local fallbacks")
    
    yield
    
    logger.info("Shutting down Eindr Microservices API")

# Create FastAPI app
app = FastAPI(
    title="Eindr Microservices API",
    description="Unified API for Eindr microservices platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Add security middleware if available
if SECURITY_AVAILABLE:
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    logger.info("Rate limiting middleware enabled")

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# JWT Authentication
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        secret_key = os.getenv("JWT_SECRET", "your-secret-key-here")
        payload = jwt.decode(credentials.credentials, secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Eindr Microservices API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "auth": "/auth",
            "customers": "/customers", 
            "reminders": "/reminders",
            "notes": "/notes",
            "ledger": "/ledger",
            "friends": "/friends",
            "chat": "/chat",
            "health": "/health"
        }
    }

# Dependency injection for Redis and health checker
def get_redis() -> Optional[RedisManager]:
    """Dependency to get Redis manager"""
    if SECURITY_AVAILABLE:
        return get_redis_manager()
    return None

def get_health() -> Optional[HealthChecker]:
    """Dependency to get health checker"""
    if SECURITY_AVAILABLE:
        return get_health_checker()
    return None

@app.get("/health")
async def health_check(health_checker: Optional[HealthChecker] = Depends(get_health)):
    """Basic health check endpoint"""
    base_health = {
        "status": "healthy",
        "service": "eindr-api",
        "version": "1.0.0",
        "ml_available": ML_AVAILABLE,
        "security_available": SECURITY_AVAILABLE
    }
    
    if health_checker:
        comprehensive_health = health_checker.get_comprehensive_health()
        base_health.update(comprehensive_health)
    
    return base_health

@app.get("/health/comprehensive")
async def comprehensive_health_check(health_checker: Optional[HealthChecker] = Depends(get_health)):
    """Comprehensive health check endpoint"""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not available")
    
    return health_checker.get_comprehensive_health()

@app.get("/health/redis")
async def redis_health_check(health_checker: Optional[HealthChecker] = Depends(get_health)):
    """Redis-specific health check"""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not available")
    
    return health_checker.check_redis_health()

@app.post("/admin/redis/reconnect")
async def force_redis_reconnect(health_checker: Optional[HealthChecker] = Depends(get_health)):
    """Force Redis reconnection (admin endpoint)"""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not available")
    
    return health_checker.force_redis_reconnect()

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "operational",
        "ml_available": ML_AVAILABLE,
        "services": {
            "auth": "available",
            "customer": "available", 
            "chat": "available" if ML_AVAILABLE else "limited",
            "stt": "available" if ML_AVAILABLE else "limited",
            "tts": "available" if ML_AVAILABLE else "limited",
            "intent": "available" if ML_AVAILABLE else "limited",
            "reminder": "available",
            "note": "available",
            "ledger": "available",
            "friend": "available",
            "history": "available",
            "scheduler": "available"
        }
    }

# ======================
# AUTH SERVICE ROUTES
# ======================

@app.post("/auth/register")
@rate_limit("auth.register") if SECURITY_AVAILABLE else lambda f: f
async def register_user(request: Request):
    """Register a new user"""
    try:
        body = await request.json()
        # Connect to actual auth service
        result = await microservice_client.register_user(body)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register_user: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
@rate_limit("auth.login") if SECURITY_AVAILABLE else lambda f: f
async def login_user(request: Request):
    """Login user"""
    try:
        body = await request.json()
        # Connect to actual auth service
        result = await microservice_client.login_user(body)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in login_user: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/auth/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information"""
    try:
        # Extract token from current_user (assuming it's the JWT payload)
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            # If no token in payload, try to get from request
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual auth service
        result = await microservice_client.get_user_info(token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user_info: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# CUSTOMER SERVICE ROUTES
# ======================

@app.get("/customers")
@rate_limit("api.read") if SECURITY_AVAILABLE else lambda f: f
async def get_customers(current_user = Depends(get_current_user)):
    """Get all customers"""
    try:
        # Extract token from current_user
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual customer service
        result = await microservice_client.get_customers(token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_customers: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customers/{customer_id}")
async def get_customer(customer_id: int, current_user = Depends(get_current_user)):
    """Get customer by ID"""
    try:
        # Extract token from current_user
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual customer service
        result = await microservice_client.get_customer(customer_id, token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_customer: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/customers")
async def create_customer(request: Request, current_user = Depends(get_current_user)):
    """Create a new customer"""
    try:
        body = await request.json()
        # Extract token from current_user
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual customer service
        result = await microservice_client.create_customer(body, token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_customer: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# REMINDER SERVICE ROUTES
# ======================

@app.get("/reminders")
async def get_reminders(current_user = Depends(get_current_user)):
    """Get all reminders for current user"""
    # Mock implementation
    return {
        "reminders": [
            {"id": 1, "title": "Meeting", "description": "Team meeting", "due_date": "2024-01-15T10:00:00Z"},
            {"id": 2, "title": "Call", "description": "Client call", "due_date": "2024-01-16T14:00:00Z"}
        ]
    }

@app.post("/reminders")
async def create_reminder(request: Request, current_user = Depends(get_current_user)):
    """Create a new reminder"""
    try:
        body = await request.json()
        return {
            "id": 123,
            "title": body.get("title"),
            "description": body.get("description"),
            "due_date": body.get("due_date"),
            "user_id": current_user.get("sub")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# NOTE SERVICE ROUTES
# ======================

@app.get("/notes")
async def get_notes(current_user = Depends(get_current_user)):
    """Get all notes for current user"""
    # Mock implementation
    return {
        "notes": [
            {"id": 1, "title": "Project Ideas", "content": "Some project ideas...", "created_at": "2024-01-01T00:00:00Z"},
            {"id": 2, "title": "Meeting Notes", "content": "Important meeting notes...", "created_at": "2024-01-02T00:00:00Z"}
        ]
    }

@app.post("/notes")
async def create_note(request: Request, current_user = Depends(get_current_user)):
    """Create a new note"""
    try:
        body = await request.json()
        return {
            "id": 123,
            "title": body.get("title"),
            "content": body.get("content"),
            "user_id": current_user.get("sub"),
            "created_at": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# LEDGER SERVICE ROUTES
# ======================

@app.get("/ledger")
async def get_ledger_entries(current_user = Depends(get_current_user)):
    """Get all ledger entries for current user"""
    # Mock implementation
    return {
        "entries": [
            {"id": 1, "description": "Salary", "amount": 5000, "type": "income", "date": "2024-01-01"},
            {"id": 2, "description": "Rent", "amount": -1500, "type": "expense", "date": "2024-01-05"}
        ]
    }

@app.post("/ledger")
async def create_ledger_entry(request: Request, current_user = Depends(get_current_user)):
    """Create a new ledger entry"""
    try:
        body = await request.json()
        return {
            "id": 123,
            "description": body.get("description"),
            "amount": body.get("amount"),
            "type": body.get("type"),
            "user_id": current_user.get("sub"),
            "date": body.get("date")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# FRIEND SERVICE ROUTES
# ======================

@app.get("/friends")
async def get_friends(current_user = Depends(get_current_user)):
    """Get all friends for current user"""
    # Mock implementation
    return {
        "friends": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "status": "active"},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "status": "pending"}
        ]
    }

@app.post("/friends")
async def add_friend(request: Request, current_user = Depends(get_current_user)):
    """Add a new friend"""
    try:
        body = await request.json()
        return {
            "id": 123,
            "name": body.get("name"),
            "email": body.get("email"),
            "user_id": current_user.get("sub"),
            "status": "pending"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# CHAT SERVICE ROUTES
# ======================

@app.post("/chat")
@rate_limit("ai.chat") if SECURITY_AVAILABLE else lambda f: f
async def chat_endpoint(request: Request, current_user = Depends(get_current_user)):
    """Chat with AI"""
    try:
        body = await request.json()
        message = body.get("message", "")
        
        if not ML_AVAILABLE:
            return {
                "response": "AI services are currently unavailable. Please try again later.",
                "conversation_id": "mock_conv_123",
                "message_id": "mock_msg_456",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        
        # Mock AI response
        return {
            "response": f"AI response to: {message}",
            "conversation_id": "mock_conv_123",
            "message_id": "mock_msg_456",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# REDIS CACHE DEMO ROUTES
# ======================

@app.get("/cache/{key}")
@rate_limit("api.read") if SECURITY_AVAILABLE else lambda f: f
async def get_cached_value(key: str, redis_manager: Optional[RedisManager] = Depends(get_redis)):
    """Get value from cache with graceful fallback"""
    if not redis_manager:
        return {
            "key": key,
            "value": None,
            "source": "redis_unavailable",
            "redis_available": False
        }
    
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
@rate_limit("api.write") if SECURITY_AVAILABLE else lambda f: f
async def set_cached_value(
    key: str, 
    request: Request,
    redis_manager: Optional[RedisManager] = Depends(get_redis)
):
    """Set value in cache with graceful fallback"""
    try:
        body = await request.json()
        value = body.get("value", "")
        ttl = body.get("ttl", 300)
        
        if not redis_manager:
            return {
                "key": key,
                "value": value,
                "ttl": ttl,
                "success": False,
                "redis_available": False,
                "message": "Redis not available"
            }
        
        success = redis_manager.set(key, value, ex=ttl)
        
        return {
            "key": key,
            "value": value,
            "ttl": ttl,
            "success": success,
            "redis_available": redis_manager.is_available
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/cache/{key}")
@rate_limit("api.delete") if SECURITY_AVAILABLE else lambda f: f
async def delete_cached_value(key: str, redis_manager: Optional[RedisManager] = Depends(get_redis)):
    """Delete value from cache"""
    if not redis_manager:
        return {
            "key": key,
            "deleted": False,
            "redis_available": False,
            "message": "Redis not available"
        }
    
    deleted_count = redis_manager.delete(key)
    
    return {
        "key": key,
        "deleted": deleted_count > 0,
        "redis_available": redis_manager.is_available
    }

# Global exception handler with Redis status
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with Redis status"""
    redis_manager = get_redis_manager() if SECURITY_AVAILABLE else None
    
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "redis_available": redis_manager.is_available if redis_manager else False,
            "security_available": SECURITY_AVAILABLE,
            "timestamp": str(time.time())
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)