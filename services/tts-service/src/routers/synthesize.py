from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel, Field
import logging
import io

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
async def synthesize_speech(request: TTSRequest):
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
            text=request.text,
            language=request.language,
            voice=request.voice,
            speed=request.speed
        )
        
        logger.info(f"TTS synthesis completed: '{request.text[:50]}...' using {result['engine_used']}")
        
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
async def synthesize_speech_json(request: TTSRequest):
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
            text=request.text,
            language=request.language,
            voice=request.voice,
            speed=request.speed
        )
        
        # In a real implementation, you would save the audio file and return URL
        audio_url = f"/audio/{hash(request.text)}.wav"  # Placeholder URL
        
        return TTSResponse(
            audio_url=audio_url,
            duration=result["estimated_duration"],
            language=request.language,
            voice=request.voice,
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


@router.post("/synthesize/json", response_model=TTSResponse)
async def synthesize_speech_json(request: TTSRequest):
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
            text=request.text,
            language=request.language,
            voice=request.voice,
            speed=request.speed
        )
        
        # In a real implementation, you would save the audio file and return URL
        audio_url = f"/audio/{hash(request.text)}.wav"  # Placeholder URL
        
        return TTSResponse(
            audio_url=audio_url,
            duration=result["estimated_duration"],
            language=request.language,
            voice=request.voice,
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

    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"}
        ]
    }
