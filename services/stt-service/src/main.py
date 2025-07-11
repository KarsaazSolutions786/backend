from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os

from .config import settings
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

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Speech-to-Text service...")
    
    try:
        # The Hugging Face Whisper model is loaded directly in the transcribe router
        logger.info("Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    logger.info("Service shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="STT Service",
    description="Speech-to-Text service with Whisper",
    version="1.0.0",
    lifespan=lifespan
)

# Set limiter on app state
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

# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Import and include routers after app is created
from .routers import transcribe
app.include_router(transcribe.router, prefix="/stt", tags=["speech-to-text"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if the Hugging Face model is loaded by importing the transcribe module
    try:
        from .routers.transcribe import WHISPER_MODEL
        model_status = "available" if WHISPER_MODEL else "unavailable"
    except:
        model_status = "unavailable"
    
    return {
        "status": "healthy",
        "service": "stt-service",
        "version": "1.0.0",
        "whisper_model": model_status,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "stt-service",
        "message": "Speech-to-Text service with Whisper is running",
        "version": "1.0.0",
        "docs": "/docs"
    }
