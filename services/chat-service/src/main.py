from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os
import sys

# Add shared modules to path

from .config import settings
from .database import init_db
from .routers import conversation
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import shared security configuration
try:
    from security_config import SecurityConfig, add_security_headers_middleware, SensitiveDataFilter
    from rate_limiting import RateLimitMiddleware
    from secure_error_handling import SecurityResponseMiddleware
    HAS_SHARED_SECURITY = True
except ImportError:
    # Fallback if shared security module not available
    HAS_SHARED_SECURITY = False
    print("Warning: Shared security module not available, using fallback configuration")

# Configure secure logging
class SecureFormatter(logging.Formatter):
    """Custom formatter that filters sensitive data"""
    def format(self, record):
        if hasattr(record, 'msg') and HAS_SHARED_SECURITY:
            record.msg = SensitiveDataFilter.filter_sensitive_data(record.msg)
        return super().format(record)

# Configure logging with security
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Apply secure formatter to all handlers
for handler in logging.root.handlers:
    handler.setFormatter(SecureFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logger = logging.getLogger(__name__)

# Secure CORS configuration
if HAS_SHARED_SECURITY:
    ALLOWED_ORIGINS = SecurityConfig.ALLOWED_ORIGINS
    logger.info(f"Using secure CORS configuration with origins: {ALLOWED_ORIGINS}")
else:
    # Fallback CORS configuration (not recommended for production)
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
    logger.warning(f"Using fallback CORS configuration: {ALLOWED_ORIGINS}")

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Chat and conversation service...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        logger.info("Chat service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize chat service: {str(e)}")
        raise
    
    yield
    
    logger.info("Chat service shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="Chat Service",
    description="Secure chat and conversation management service",
    version="1.0.0",
    lifespan=lifespan
)

# Apply security configurations
if HAS_SHARED_SECURITY:
    # Add security middlewares in proper order
    app.add_middleware(SecurityResponseMiddleware, environment=settings.ENVIRONMENT)
    app.add_middleware(RateLimitMiddleware, default_endpoint="api.chat")
    add_security_headers_middleware(app)
    
    # Use shared security configuration for CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token", "X-Request-ID"]
    )
    logger.info("Security middleware enabled")
else:
    # Fallback CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"]
    )
    logger.warning("Using fallback security configuration")

# Rate limiting exception handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    response = JSONResponse(
        status_code=429, 
        content={"detail": "Rate limit exceeded", "retry_after": exc.retry_after}
    )
    return response

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Store limiter on app state
app.state.limiter = limiter

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "chat-service",
        "version": "1.0.0",
        "security": "enabled" if HAS_SHARED_SECURITY else "fallback"
    }

# Include routers
app.include_router(conversation.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
