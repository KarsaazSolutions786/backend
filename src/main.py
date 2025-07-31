"""
Main application entry point for Railway deployment
This serves as a unified API gateway for all microservices
"""

import os
import sys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import time
import jwt
from typing import Optional

# Import microservice client
from .services.microservice_client import microservice_client

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

# JWT Authentication
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        secret_key = os.getenv("JWT_SECRET", "your-secret-key-here")
        payload = jwt.decode(credentials.credentials, secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Eindr Microservices API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "auth": "/auth",
            "customers": "/customers", 
            "reminders": "/reminders",
            "notes": "/notes",
            "ledger": "/ledger",
            "friends": "/friends",
            "chat": "/chat",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "eindr-api",
        "version": "1.0.0",
        "ml_available": ML_AVAILABLE
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

# ======================
# AUTH SERVICE ROUTES
# ======================

@app.post("/auth/register")
async def register_user(request: Request):
    """Register a new user"""
    try:
        body = await request.json()
        # Connect to actual auth service
        result = await microservice_client.register_user(body)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register_user: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login_user(request: Request):
    """Login user"""
    try:
        body = await request.json()
        # Connect to actual auth service
        result = await microservice_client.login_user(body)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in login_user: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/auth/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information"""
    try:
        # Extract token from current_user (assuming it's the JWT payload)
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            # If no token in payload, try to get from request
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual auth service
        result = await microservice_client.get_user_info(token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user_info: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# CUSTOMER SERVICE ROUTES
# ======================

@app.get("/customers")
async def get_customers(current_user = Depends(get_current_user)):
    """Get all customers"""
    try:
        # Extract token from current_user
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual customer service
        result = await microservice_client.get_customers(token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_customers: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customers/{customer_id}")
async def get_customer(customer_id: int, current_user = Depends(get_current_user)):
    """Get customer by ID"""
    try:
        # Extract token from current_user
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual customer service
        result = await microservice_client.get_customer(customer_id, token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_customer: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/customers")
async def create_customer(request: Request, current_user = Depends(get_current_user)):
    """Create a new customer"""
    try:
        body = await request.json()
        # Extract token from current_user
        token = current_user.get("token") if isinstance(current_user, dict) else None
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Connect to actual customer service
        result = await microservice_client.create_customer(body, token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_customer: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# ======================
# REMINDER SERVICE ROUTES
# ======================
# Note: Reminder service endpoints are handled by the dedicated reminder microservice

# ======================
# NOTE SERVICE ROUTES
# ======================
# Note: Note service endpoints are handled by the dedicated note microservice

# ======================
# LEDGER SERVICE ROUTES
# ======================
# Note: Ledger service endpoints are handled by the dedicated ledger microservice

# ======================
# FRIEND SERVICE ROUTES
# ======================
# Note: Friend service endpoints are handled by the dedicated friend microservice

# ======================
# CHAT SERVICE ROUTES
# ======================

@app.post("/chat")
async def chat_endpoint(request: Request, current_user = Depends(get_current_user)):
    """Chat with AI"""
    try:
        body = await request.json()
        message = body.get("message", "")
        
        if not ML_AVAILABLE:
            return {
                "response": "AI services are currently unavailable. Please try again later.",
                "conversation_id": "mock_conv_123",
                "message_id": "mock_msg_456",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        
        # Mock AI response
        return {
            "response": f"AI response to: {message}",
            "conversation_id": "mock_conv_123",
            "message_id": "mock_msg_456",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)