from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from .config import settings
from .routers import classify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Intent classification service...")
    
    try:
        # Initialize Intent service
        from .services.intent_service import initialize_intent_service
        intent_service = initialize_intent_service()
        await intent_service.load_model()
        
        logger.info("Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    logger.info("Service shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="Intent Service",
    description="Intent classification service with MiniLM",
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
app.include_router(classify.router, prefix="/intent", tags=["intent-classification"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .services.intent_service import get_intent_service
    intent_service = get_intent_service()
    
    model_status = "available" if intent_service and intent_service.is_available() else "unavailable"
    
    return {
        "status": "healthy",
        "service": "intent-service",
        "version": "1.0.0",
        "model": model_status,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "intent-service",
        "message": "Intent classification service with MiniLM is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "intent-service",
        "message": "Intent classification service with MiniLM is running",
        "version": "1.0.0",
        "docs": "/docs"
    }
