from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from .config import settings
from .routers import synthesize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Text-to-Speech service...")
    
    try:
        # Initialize TTS service
        from .services.tts_service import initialize_tts_service
        tts_service = initialize_tts_service()
        await tts_service.load_model()
        
        logger.info("Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    logger.info("Service shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech service with Coqui TTS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Include routers
app.include_router(synthesize.router, prefix="/tts", tags=["text-to-speech"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .services.tts_service import get_tts_service
    tts_service = get_tts_service()
    
    engine_status = "available" if tts_service and tts_service.is_available() else "unavailable"
    
    return {
        "status": "healthy",
        "service": "tts-service",
        "version": "1.0.0",
        "tts_engines": engine_status,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "tts-service",
        "message": "Text-to-Speech service with Coqui TTS is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "tts-service",
        "message": "Text-to-Speech service with Coqui TTS is running",
        "version": "1.0.0",
        "docs": "/docs"
    }
