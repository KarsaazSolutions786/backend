"""
Main application entry point for Railway deployment
This serves as a unified API gateway for all microservices
"""

import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../shared'))

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

# Create FastAPI app
app = FastAPI(
    title="Eindr Microservices API",
    description="Unified API for Eindr microservices platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Eindr Microservices API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "eindr-api",
        "version": "1.0.0"
    }

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 