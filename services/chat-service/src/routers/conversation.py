from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import httpx
import json
from ..services.bloom_service import get_bloom_service
# Use secure shared authentication 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../shared'))

try:
    from simple_auth import get_current_customer_id
except ImportError:
    # Fallback to local secure implementation
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi import Depends, HTTPException, status
    import jwt
    
    def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
        """Secure local implementation as fallback"""
        try:
            token = credentials.credentials
            secret_key = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
            
            # Validate production environment
            if os.getenv("ENVIRONMENT") == "production" and secret_key in [
                "your-secret-key-here", 
                "your-secret-key-here-change-in-production",
                "eindr-super-secret-key-change-in-production-123456789"
            ]:
                raise ValueError("Production environment requires a secure SECRET_KEY")
            
            payload = jwt.decode(
                token, 
                secret_key, 
                algorithms=["HS256"],
                audience="eindr-api",
                issuer="eindr-issuer",
                options={"verify_signature": True, "verify_exp": True, "verify_aud": True, "verify_iss": True}
            )
            customer_id = payload.get("sub")
            if customer_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing subject"
                )
            return int(customer_id)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from fastapi.responses import JSONResponse

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    customer_id: str
    context: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = Field(0.8, ge=0.1, le=2.0)
    max_length: Optional[int] = Field(150, ge=10, le=500)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None

# In-memory conversation storage (in production, use a database)
conversation_store: Dict[str, List[Dict]] = {}

@router.post("/chat")
@limiter.limit("100/minute") # Apply rate limiting using the decorator
async def chat_endpoint(
    request_data: ChatRequest, 
    request: Request,
    current_customer_id: int = Depends(get_current_customer_id)
):
    """Main chat endpoint for AI conversations using BLOOM-560M"""
    try:
        # Get BLOOM service
        bloom_service = get_bloom_service()
        
        # Get or create conversation history
        conversation_id = request_data.conversation_id or f"conv_{request_data.customer_id}_{int(datetime.utcnow().timestamp())}"
        
        if conversation_id not in conversation_store:
            conversation_store[conversation_id] = []
        
        conversation_history = conversation_store[conversation_id]
        
        # Generate AI response using BLOOM
        ai_response = bloom_service.chat_response(
            message=request_data.message,
            conversation_history=conversation_history,
            customer_id=request_data.customer_id
        )
        
        # Store the conversation
        conversation_history.append({
            "role": "user",
            "content": request_data.message,
            "timestamp": datetime.utcnow()
        })
        
        conversation_history.append({
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.utcnow()
        })
        
        # Keep only last 10 messages to prevent context overflow
        if len(conversation_history) > 10:
            conversation_store[conversation_id] = conversation_history[-10:]
        
        message_id = f"msg_{int(datetime.utcnow().timestamp())}"
        
        # Get model info for response
        model_info = bloom_service.get_model_info()
        
        logger.info(f"Chat response generated for conversation {conversation_id}: {ai_response[:50]}...")
        
        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            message_id=message_id,
            timestamp=datetime.utcnow(),
            context=request_data.context,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request"
        )

@router.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    if conversation_id in conversation_store:
        return {
            "conversation_id": conversation_id,
            "messages": conversation_store[conversation_id],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    else:
        return {
            "conversation_id": conversation_id,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation history"""
    if conversation_id in conversation_store:
        del conversation_store[conversation_id]
    return {"message": "Conversation deleted successfully"}

@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded BLOOM model"""
    try:
        bloom_service = get_bloom_service()
        return bloom_service.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )

@router.post("/generate")
async def generate_text(request_data: ChatRequest):
    """Direct text generation endpoint"""
    try:
        bloom_service = get_bloom_service()
        
        response = bloom_service.generate_response(
            prompt=request_data.message,
            max_length=request_data.max_length or 150,
            temperature=request_data.temperature or 0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
        
        return {
            "generated_text": response,
            "model_info": bloom_service.get_model_info(),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate text"
        )
