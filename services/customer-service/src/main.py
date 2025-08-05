from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os

from .config import settings
from .database import init_db
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://yourdomain.com").split(",")

# Create limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Customer management service...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # HTTP client removed - using direct database queries instead
        logger.info("Direct database access configured")
        
        logger.info("Customer service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize customer service: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    try:
        from .routers.customers import close_http_client
        await close_http_client()
        logger.info("HTTP client closed successfully")
    except Exception as e:
        logger.error(f"Error closing HTTP client: {e}")
    
    logger.info("Customer service shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="Customer Service",
    description="Customer profile and preferences management for Eindr",
    version="1.0.0",
    lifespan=lifespan
)

# Set limiter state
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # No wildcards!
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Import routers after app creation to avoid circular imports
from .routers import customers

# Include routers with customers endpoints
app.include_router(customers.router, prefix="/customers", tags=["customers"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "customer-service",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "customer-service",
        "message": "Customer profile and preferences management for Eindr is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "customers": "/customers",
            "health": "/health",
            "docs": "/docs"
        }
    }
