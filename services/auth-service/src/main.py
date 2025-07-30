import uvicorn
import os
from contextlib import asynccontextmanager

# Import enhanced services from the shared module
from shared.enhanced_services import (
    api_service,
    db_service,
    logger,
    monitoring_service,
    error_handler,
    config_service,
    security_config
)

# Import local routers
from .routers import auth

# Get the FastAPI app instance from the EnhancedAPIService
app = api_service.get_app()

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app_instance: api_service.FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Auth Service is starting up...")
    try:
        # Connect to the database
        await db_service.connect()
        logger.info("Database connection established.")
        
        # Start monitoring background tasks if enabled
        if monitoring_service.is_enabled():
            monitoring_service.start_system_monitoring()
            logger.info("System monitoring has started.")
            
    except Exception as e:
        logger.error(f"Critical error during startup: {e}", exc_info=True)
        # In a production scenario, you might want to prevent the service from starting
        raise

    yield

    logger.info("Auth Service is shutting down...")
    # Disconnect from the database
    await db_service.disconnect()
    logger.info("Database connection closed.")
    
    # Stop monitoring
    if monitoring_service.is_enabled():
        monitoring_service.stop_system_monitoring()
        logger.info("System monitoring has stopped.")

# Assign the lifespan manager to the app
app.router.lifespan_context = lifespan

# --- Router Inclusion ---
# Include the authentication routes
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# --- Root and Health Check Endpoints ---
# These are now provided by the EnhancedAPIService, but we can override or add more.

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint providing basic service information."""
    return {
        "service": config_service.get_config('service').name,
        "version": config_service.get_config('service').version,
        "status": "healthy",
        "environment": config_service.environment
    }

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "service": "auth-service",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "auth-service"}

if __name__ == "__main__":
    # Get server config from the enhanced config service
    api_conf = config_service.get_config('api')
    
    logger.info(f"Starting Auth Service on {api_conf.host}:{api_conf.port}")
    
    uvicorn.run(
        "src.main:app",
        host=api_conf.host,
        port=api_conf.port,
        reload=api_conf.reload,
        log_level=logger.config.level.lower() if logger.config else 'info'
    )