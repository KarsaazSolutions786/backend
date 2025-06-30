from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from typing import Optional, Dict, Any
from pydantic import BaseModel
import logging
import tempfile
import os

router = APIRouter()
logger = logging.getLogger(__name__)

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration: float
    segments: Optional[list] = None

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
    language: str = Form("auto"),
    include_segments: bool = Form(False)
):
    """Transcribe audio file to text using Whisper"""
    try:
        # Get Whisper service
        from ..services.whisper_service import get_whisper_service
        whisper_service = get_whisper_service()
        
        if not whisper_service or not whisper_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Whisper model not available"
            )
        
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
            result = await whisper_service.transcribe(temp_file_path, language)
            
            logger.info(f"Transcribed for user {user_id}: '{result['text'][:100]}...' (lang: {result['language']}, conf: {result['confidence']:.2f})")
            
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                confidence=result["confidence"],
                duration=result["duration"],
                segments=result["segments"] if include_segments else None
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
    from ..services.whisper_service import get_whisper_service
    whisper_service = get_whisper_service()
    
    if whisper_service and whisper_service.is_available():
        model_info = whisper_service.get_model_info()
        return {
            "status": "healthy",
            "service": "stt-service",
            "whisper_model": "available",
            "model_info": model_info
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
    from ..services.whisper_service import get_whisper_service
    whisper_service = get_whisper_service()
    
    if not whisper_service or not whisper_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper model not loaded"
        )
    
    return whisper_service.get_model_info()

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
    from ..services.whisper_service import get_whisper_service
    whisper_service = get_whisper_service()
    
    if whisper_service and whisper_service.is_available():
        model_info = whisper_service.get_model_info()
        return {
            "status": "healthy",
            "service": "stt-service",
            "whisper_model": "available",
            "model_info": model_info
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
    from ..services.whisper_service import get_whisper_service
    whisper_service = get_whisper_service()
    
    if not whisper_service or not whisper_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper model not loaded"
        )
    
    return whisper_service.get_model_info()

