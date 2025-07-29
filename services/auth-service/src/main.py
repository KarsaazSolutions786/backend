from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import sys

from .config import settings
from .database import init_db
from .routers import auth
from .utils.logger import logger

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../shared'))

# Import enhanced Redis manager and security modules
try:
    from redis_manager import get_redis_manager, RedisManager
    from health_check import get_health_checker, HealthChecker
    from rate_limiting import RateLimitMiddleware, rate_limit
    from csrf_protection import csrf_protection
    SECURITY_AVAILABLE = True
    logger.info("Enhanced security modules loaded successfully")
except ImportError as e:
    logger.warning(f"Security modules not available: {e}")
    SECURITY_AVAILABLE = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting Auth Service...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Initialize Redis manager if available
        if SECURITY_AVAILABLE:
            redis_manager = get_redis_manager()
            health_status = redis_manager.get_status()
            
            if health_status["available"]:
                logger.info("Redis connection established successfully")
            else:
                logger.warning("Redis not available - service will use local fallbacks")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Auth Service...")

# Create FastAPI app
app = FastAPI(
    title="Eindr Auth Service",
    description="Authentication and authorization service for Eindr",
    version="1.0.0",
    lifespan=lifespan
)

# Add security middleware if available
if SECURITY_AVAILABLE:
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    logger.info("Rate limiting middleware enabled")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "service": "auth-service",
        "version": "1.0.0",
        "status": "healthy"
    }

# Dependency injection for Redis and health checker
def get_redis() -> RedisManager:
    """Dependency to get Redis manager"""
    if SECURITY_AVAILABLE:
        return get_redis_manager()
    return None

def get_health() -> HealthChecker:
    """Dependency to get health checker"""
    if SECURITY_AVAILABLE:
        return get_health_checker()
    return None

@app.get("/health")
async def health_check(health_checker: HealthChecker = Depends(get_health)):
    """Health check endpoint."""
    base_health = {
        "status": "healthy", 
        "service": "auth-service",
        "security_available": SECURITY_AVAILABLE
    }
    
    if health_checker:
        comprehensive_health = health_checker.get_comprehensive_health()
        base_health.update(comprehensive_health)
    
    return base_health

@app.get("/health/comprehensive")
async def comprehensive_health_check(health_checker: HealthChecker = Depends(get_health)):
    """Comprehensive health check endpoint"""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not available")
    
    return health_checker.get_comprehensive_health()

@app.get("/health/redis")
async def redis_health_check(health_checker: HealthChecker = Depends(get_health)):
    """Redis-specific health check"""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not available")
    
    return health_checker.check_redis_health()

if __name__ == "__main__":
    # Get port from environment variable (Railway sets PORT)
    port = int(os.getenv("PORT", settings.PORT))
    host = os.getenv("HOST", settings.HOST)
    
    print(f"Starting Auth Service on {host}:{port}")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=settings.DEBUG
    )