from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging
import io
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
# from ..database import get_db  # Not needed for TTS service
# from ..services.tts_service import TTS_Service  # Temporarily disabled - using function imports
# from ..models.tts_request import TTSRequest, TTSResponse, VoiceProfile  # Using local models
# Use secure shared authentication 
import sys
import os

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
                options={"verify_signature": True, "verify_exp": True}
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

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="en")
    voice: Optional[str] = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

class TTSResponse(BaseModel):
    audio_url: str
    duration: float
    language: str
    voice: str
    engine_used: str

@router.post("/synthesize")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute
async def synthesize_speech(
    request_data: TTSRequest, 
    request: Request, 
    current_customer_id: int = Depends(get_current_customer_id)
):
    """Convert text to speech using Coqui TTS with fallbacks"""
    try:
        # Get TTS service
        from ..services.tts_service import get_tts_service
        tts_service = get_tts_service()
        
        if not tts_service or not tts_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No TTS engine available"
            )
        
        # Synthesize speech
        result = await tts_service.synthesize(
            text=request_data.text,
            language=request_data.language,
            voice=request_data.voice,
            speed=request_data.speed
        )
        
        logger.info(f"TTS synthesis completed: '{request_data.text[:50]}...' using {result['engine_used']}")
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(result["audio_data"]),
            media_type=result["content_type"],
            headers={
                "Content-Disposition": f"attachment; filename=speech.{'mp3' if result['content_type'] == 'audio/mpeg' else 'wav'}",
                "X-Engine-Used": result["engine_used"],
                "X-Text-Length": str(result["text_length"])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to synthesize speech"
        )

@router.post("/synthesize/json", response_model=TTSResponse)
async def synthesize_speech_json(request_data: TTSRequest):
    """Convert text to speech and return metadata (for pipeline use)"""
    try:
        from ..services.tts_service import get_tts_service
        tts_service = get_tts_service()
        
        if not tts_service or not tts_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No TTS engine available"
            )
        
        result = await tts_service.synthesize(
            text=request_data.text,
            language=request_data.language,
            voice=request_data.voice,
            speed=request_data.speed
        )
        
        # In a real implementation, you would save the audio file and return URL
        audio_url = f"/audio/{hash(request_data.text)}.wav"  # Placeholder URL
        
        return TTSResponse(
            audio_url=audio_url,
            duration=result["estimated_duration"],
            language=request_data.language,
            voice=request_data.voice,
            engine_used=result["engine_used"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to synthesize speech"
        )

@router.get("/voices")
async def get_available_voices():
    """Get list of available voices"""
    from ..services.tts_service import get_tts_service
    tts_service = get_tts_service()
    
    if tts_service:
        voices = tts_service.get_available_voices()
    else:
        voices = [{"id": "default", "name": "Default", "language": "en"}]
    
    return {"voices": voices}

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    from ..services.tts_service import get_tts_service
    tts_service = get_tts_service()
    
    if tts_service:
        languages = tts_service.get_supported_languages()
    else:
        languages = [{"code": "en", "name": "English"}]
    
    return {"languages": languages}

@router.get("/engines")
async def get_engine_status():
    """Get status of all TTS engines"""
    from ..services.tts_service import get_tts_service
    tts_service = get_tts_service()
    
    if tts_service:
        return tts_service.get_engine_status()
    else:
        return {
            "coqui": "unavailable",
            "gtts": "unavailable",
            "pyttsx3": "unavailable",
            "primary_engine": "none"
        }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from ..services.tts_service import get_tts_service
    tts_service = get_tts_service()
    
    if tts_service and tts_service.is_available():
        engine_status = tts_service.get_engine_status()
        return {
            "status": "healthy",
            "service": "tts-service",
            "engines": engine_status
        }
    else:
        return {
            "status": "degraded",
            "service": "tts-service",
            "engines": {"message": "No engines available"}
        }
