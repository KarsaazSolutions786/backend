from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://yourdomain.com").split(",")

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AI Pipeline Service", 
    version="1.0.0",
    description="Orchestrates AI services: STT, Intent Classification, Chat, and TTS"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # No wildcards!
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

# Set limiter on app state
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

# Import and include routers after app is created
from .routers import pipeline
app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ai-pipeline-service",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AI Pipeline Service",
        "version": "1.0.0",
        "description": "Orchestrates AI services for speech-to-text, intent classification, chat, and text-to-speech",
        "endpoints": {
            "health": "/health",
            "services_status": "/pipeline/services/status",
            "audio_pipeline": "/pipeline/audio-pipeline",
            "text_pipeline": "/pipeline/text-pipeline",
            "synthesize_audio": "/pipeline/synthesize-audio"
        }
    } 
 
 