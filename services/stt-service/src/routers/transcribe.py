from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, Request
from typing import Optional, Dict, Any
from pydantic import BaseModel
import logging
import tempfile
import os

# Import openai-whisper library
import whisper

# Import config
from ..config import settings
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from fastapi.responses import JSONResponse
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

# Load the Whisper model
WHISPER_MODEL = whisper.load_model("tiny")

router = APIRouter()
logger = logging.getLogger(__name__)

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration: float
    segments: Optional[list] = None

@router.post("/transcribe")
async def transcribe(
    audio: UploadFile, 
    request: Request,
    language: str = Form(default="auto"),
    include_segments: bool = Form(default=False),
    current_customer_id: int = Depends(get_current_customer_id)
):
    """Transcribe audio file to text using OpenAI Whisper model"""
    # Get limiter from app state to avoid circular import
    limiter = request.app.state.limiter
    
    # Apply rate limiting
    await limiter.check_request_and_update(request)
    
    try:
        # Validate file type
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an audio file"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe audio using Whisper
            result = WHISPER_MODEL.transcribe(
                temp_file_path,
                language=language if language != "auto" else None,
                task="transcribe"
            )
            
            # Extract transcription text
            transcription = result["text"]
            
            # Get language info
            detected_language = result.get("language", "en")
            
            # Calculate confidence from segments
            segments = result.get("segments", [])
            avg_confidence = 0.85  # Default confidence
            if segments:
                confidences = []
                for segment in segments:
                    # Estimate confidence from no_speech_prob (inverse relationship)
                    no_speech_prob = segment.get("no_speech_prob", 0.5)
                    confidence = 1.0 - no_speech_prob
                    confidences.append(confidence)
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
            
            # Calculate duration
            duration = 1.0
            if segments:
                last_segment = segments[-1]
                duration = last_segment.get("end", 1.0)
            
            logger.info(f"Transcribed for user {current_customer_id}: '{transcription[:100]}...' (lang: {detected_language}, conf: {avg_confidence:.2f})")
            
            return TranscriptionResponse(
                text=transcription.strip(),
                language=detected_language,
                confidence=avg_confidence,
                duration=duration,
                segments=segments if include_segments else None
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transcribe audio"
        )

@router.post("/transcribe/streaming")
async def transcribe_streaming():
    """Real-time streaming transcription (placeholder for future implementation)"""
    return {
        "message": "Streaming transcription not implemented yet",
        "status": "planned"
    }

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages for Whisper"""
    return {
        "languages": [
            {"code": "auto", "name": "Auto-detect"},
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "nl", "name": "Dutch"},
            {"code": "pl", "name": "Polish"},
            {"code": "tr", "name": "Turkish"}
        ]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    if WHISPER_MODEL:
        return {
            "status": "healthy",
            "service": "stt-service",
            "whisper_model": "available",
            "model_info": {
                "model_name": "openai/whisper-tiny (local)",
                "device": "cpu", # Whisper runs on CPU
                "status": "loaded"
            }
        }
    else:
        return {
            "status": "degraded",
            "service": "stt-service",
            "whisper_model": "unavailable"
        }

@router.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    if not WHISPER_MODEL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper model not loaded"
        )
    
    return {
        "model_name": "openai/whisper-tiny (local)",
        "device": "cpu", # Whisper runs on CPU
        "status": "loaded",
        "model_type": "openai_whisper",
        "parameters": "39M",
        "languages_supported": 99,
        "model_path": "tiny" # Whisper model is built-in
    }

