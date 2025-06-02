from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from contextlib import asynccontextmanager
import os

# Import core modules
from core.config import settings
from utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    logger.info("Starting Eindr Backend...")
    
    # Initialize services only if not in minimal mode
    if not os.getenv("MINIMAL_MODE", "false").lower() == "true":
        try:
            logger.info("Loading AI models...")
            
            # Import services for initialization
            from services.stt_service import SpeechToTextService
            from services.tts_service import TextToSpeechService
            from services.intent_service import IntentService
            from services.chat_service import ChatService
            from core.dependencies import set_services
            from core.scheduler import scheduler
            
            # Initialize STT service
            logger.info("Initializing Speech-to-Text service...")
            stt_service = SpeechToTextService()
            
            # Initialize TTS service
            logger.info("Initializing Text-to-Speech service...")
            tts_service = TextToSpeechService()
            
            # Initialize Intent service
            logger.info("Initializing Intent classification service...")
            intent_service = IntentService()
            
            # Initialize Chat service
            logger.info("Initializing Chat service...")
            chat_service = ChatService()
            
            # Set services in dependencies module
            set_services(stt_service, tts_service, intent_service, chat_service)
            
            # Start scheduler
            scheduler.start()
            logger.info("All services initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            logger.warning("Continuing with limited functionality...")
            # Don't raise - allow app to start with basic functionality
    else:
        logger.info("Running in minimal mode - skipping AI service initialization")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Eindr Backend...")
    try:
        if not os.getenv("MINIMAL_MODE", "false").lower() == "true":
            from core.scheduler import scheduler
            scheduler.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Eindr - AI-Powered Reminder App",
    description="Backend API for Eindr reminder application with AI capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include essential routers only
from api import stt, ledger

# Essential APIs for voice-to-database pipeline
app.include_router(stt.router, prefix="/api/v1/stt", tags=["Speech-to-Text"])
app.include_router(ledger.router, prefix="/api/v1/ledger", tags=["Ledger"])

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Welcome to Eindr API",
        "version": "1.0.0",
        "status": "running",
        "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Basic health check that doesn't depend on AI services
        health_status = {
            "status": "healthy",
            "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
            "services": {
                "api": True,
                "database": False  # Will be updated after DB check
            }
        }
        
        # Check database connection
        try:
            from connect_db import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            health_status["services"]["database"] = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_status["services"]["database"] = False
        
        # Check AI services if available
        if not os.getenv("MINIMAL_MODE", "false").lower() == "true":
            try:
                from core.dependencies import get_stt_service, get_tts_service, get_intent_service, get_chat_service
                health_status["services"].update({
                    "stt": get_stt_service() is not None,
                    "tts": get_tts_service() is not None,
                    "intent": get_intent_service() is not None,
                    "chat": get_chat_service() is not None
                })
            except ImportError:
                # Services not initialized yet
                pass
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local"
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 