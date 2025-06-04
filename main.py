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
    
    # Always try to initialize basic services for API functionality
    try:
        logger.info("Initializing services...")
        
        # Import services for initialization
        from services.stt_service import SpeechToTextService
        from services.tts_service import TextToSpeechService
        from services.intent_service import IntentService
        from services.chat_service import ChatService
        from core.dependencies import set_services
        from core.scheduler import scheduler
        
        # Initialize services (they have their own fallback mechanisms)
        logger.info("Initializing Speech-to-Text service...")
        stt_service = SpeechToTextService()
        
        logger.info("Initializing Text-to-Speech service...")
        tts_service = TextToSpeechService()
        
        logger.info("Initializing Intent classification service...")
        intent_service = IntentService()
        
        logger.info("Initializing Chat service...")
        chat_service = ChatService()
        
        # Set services in dependencies module
        set_services(stt_service, tts_service, intent_service, chat_service)
        
        # Only start scheduler if not in minimal mode (to avoid heavy background tasks)
        if not os.getenv("MINIMAL_MODE", "false").lower() == "true":
            scheduler.start()
            logger.info("Scheduler started")
        else:
            logger.info("Scheduler skipped in minimal mode")
            
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        logger.warning("Some services may not be available, but basic functionality should work")
        # Don't raise - allow app to start with limited functionality
    
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

# Import routers after app creation to avoid circular imports
from api import auth, reminders, notes, ledger, friends, stt, users, embeddings, history, intent_processor

# Include ALL routers - Full API functionality
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(reminders.router, prefix="/api/v1/reminders", tags=["Reminders"])
app.include_router(notes.router, prefix="/api/v1/notes", tags=["Notes"])
app.include_router(ledger.router, prefix="/api/v1/ledger", tags=["Ledger"])
app.include_router(friends.router, prefix="/api/v1/friends", tags=["Friends"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"])
app.include_router(history.router, prefix="/api/v1/history", tags=["History"])
app.include_router(stt.router, prefix="/api/v1/stt", tags=["Speech-to-Text"])
app.include_router(intent_processor.router, prefix="/api/v1/intent-processor", tags=["Intent Processing"])

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Welcome to Eindr API",
        "version": "1.0.0",
        "status": "running",
        "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
        "mode": "minimal" if os.getenv("MINIMAL_MODE", "false").lower() == "true" else "full"
    }

@app.get("/health")
async def health_check_endpoint():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 