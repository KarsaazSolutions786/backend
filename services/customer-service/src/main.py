from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from .config import settings
from .database import init_db
from .routers import customers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Customer management service...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        logger.info("Customer service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize customer service: {e}")
        raise
    
    yield
    
    logger.info("Customer service shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="Customer Service",
    description="Customer profile and preferences management for Eindr",
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

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

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
