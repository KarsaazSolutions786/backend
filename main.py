from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from contextlib import asynccontextmanager
import os
import datetime

# Import core modules
from core.config import settings
from utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    logger.info("Starting Eindr Backend...")
    
    # Check if we're in minimal mode or Railway environment
    is_minimal_mode = os.getenv("MINIMAL_MODE", "false").lower() == "true"
    is_railway_env = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    # Force minimal mode in Railway
    if is_railway_env:
        is_minimal_mode = True
        logger.info("Railway environment detected - forcing minimal mode")
    
    logger.info(f"Running in {'minimal' if is_minimal_mode else 'full'} mode")
    
    # Configure PyTorch memory settings only if needed
    if not is_minimal_mode:
        try:
            from core.torch_config import configure_torch_memory
            configure_torch_memory()
        except ImportError:
            logger.warning("PyTorch configuration not available, skipping memory optimization")
    
    # Always try to initialize basic services for API functionality
    try:
        logger.info("Initializing services...")
        
        # Import services for initialization - use lighter services in minimal mode
        from services.stt_service import SpeechToTextService
        from services.tts_service import TextToSpeechService
        from services.chat_service import ChatService
        from core.dependencies import set_services
        
        # Always use lightweight intent service in Railway or minimal mode
        if is_minimal_mode or is_railway_env:
            logger.info("Using lightweight intent service (minimal/Railway mode)")
            from services.intent_service import IntentService
            intent_service = IntentService()
        else:
            logger.info("Running in full mode - attempting to use PyTorch intent service")
            try:
                # Import PyTorch service only when not in Railway/minimal mode
                from services.pytorch_intent_service import PyTorchIntentService
                intent_service = PyTorchIntentService()
            except Exception as e:
                logger.warning(f"Failed to initialize PyTorch intent service: {e}")
                logger.info("Falling back to lightweight intent service")
                from services.intent_service import IntentService
                intent_service = IntentService()
        
        # Initialize other services (they have their own fallback mechanisms)
        logger.info("Initializing Speech-to-Text service...")
        stt_service = SpeechToTextService()
        
        logger.info("Initializing Text-to-Speech service...")
        tts_service = TextToSpeechService()
        
        logger.info("Initializing Chat service...")
        chat_service = ChatService()
        
        # Set services in dependencies module
        set_services(stt_service, tts_service, intent_service, chat_service)
        
        # Only start scheduler if not in minimal mode (to avoid heavy background tasks)
        if not is_minimal_mode and not is_railway_env:
            try:
                from core.scheduler import scheduler
                scheduler.start()
                logger.info("Scheduler started")
            except Exception as e:
                logger.warning(f"Failed to start scheduler: {e}")
        else:
            logger.info("Scheduler skipped in minimal/Railway mode")
            
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        logger.warning("Some services may not be available, but basic functionality should work")
        # Don't raise - allow app to start with limited functionality
    
    yield
    
    # Cleanup
    logger.info("Shutting down Eindr Backend...")
    try:
        if not is_minimal_mode and not is_railway_env:
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
from api import auth, reminders, notes, ledger, friends, stt, users, embeddings, history, intent_processor, ai_pipeline

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
app.include_router(ai_pipeline.router, prefix="/api/v1/ai-pipeline", tags=["AI Pipeline"])

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Welcome to Eindr API",
        "version": "1.0.0",
        "status": "running",
        "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
        "mode": "minimal" if (os.getenv("MINIMAL_MODE", "false").lower() == "true" or os.getenv("RAILWAY_ENVIRONMENT")) else "full"
    }

@app.get("/health")
async def health_check_endpoint():
    """Enhanced health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "environment": os.getenv("RAILWAY_ENVIRONMENT", "local"),
            "mode": "minimal" if (os.getenv("MINIMAL_MODE", "false").lower() == "true" or os.getenv("RAILWAY_ENVIRONMENT")) else "full"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        server_header=False,
        proxy_headers=True
    ) 