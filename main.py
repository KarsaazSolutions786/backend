from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from contextlib import asynccontextmanager

# Import services for initialization
from services.stt_service import SpeechToTextService
from services.tts_service import TextToSpeechService
from services.intent_service import IntentService
from services.chat_service import ChatService
from core.config import settings
from core.scheduler import scheduler
from core.dependencies import set_services
from utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    logger.info("Starting Eindr Backend...")
    
    # Initialize AI services
    try:
        logger.info("Loading AI models...")
        stt_service = SpeechToTextService()
        tts_service = TextToSpeechService()
        intent_service = IntentService()
        chat_service = ChatService()
        
        # Set services in dependencies module
        set_services(stt_service, tts_service, intent_service, chat_service)
        
        # Start scheduler
        scheduler.start()
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down Eindr Backend...")
    scheduler.shutdown()

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
from api import auth, reminders, notes, ledger, friends, users, stt

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(reminders.router, prefix="/api/v1/reminders", tags=["Reminders"])
app.include_router(notes.router, prefix="/api/v1/notes", tags=["Notes"])
app.include_router(ledger.router, prefix="/api/v1/ledger", tags=["Ledger"])
app.include_router(friends.router, prefix="/api/v1/friends", tags=["Friends"])
app.include_router(stt.router, prefix="/api/v1/stt", tags=["Speech-to-Text"])

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Welcome to Eindr API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from core.dependencies import get_stt_service, get_tts_service, get_intent_service, get_chat_service
    
    return {
        "status": "healthy",
        "services": {
            "stt": get_stt_service() is not None,
            "tts": get_tts_service() is not None,
            "intent": get_intent_service() is not None,
            "chat": get_chat_service() is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 