from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
import aiofiles
import os
import io
from pathlib import Path
from firebase_auth import verify_firebase_token
from core.dependencies import get_stt_service, get_tts_service, get_intent_service, get_chat_service
from utils.logger import logger
from core.config import settings

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Transcribe uploaded WAV audio file to text using Coqui STT.
    
    Requirements:
    - File format: WAV (.wav)
    - Sample rate: 16 kHz
    - Channels: Mono (1 channel)
    - Bit depth: 16-bit PCM
    """
    try:
        # Get current user ID from Firebase token
        current_user_id = current_user["uid"]
        
        # Validate file type
        if not audio_file.content_type:
            raise HTTPException(
                status_code=400, 
                detail="File content type not specified"
            )
        
        # Check if it's an audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an audio file, got {audio_file.content_type}"
            )
        
        # Check file extension
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format '{file_ext}'. Supported formats: {', '.join(settings.SUPPORTED_AUDIO_FORMATS)}"
            )
        
        # Check file size
        if audio_file.size and audio_file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Create unique temporary file path
        temp_filename = f"temp_{current_user_id}_{audio_file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
        
        # Save uploaded file temporarily
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded file: {str(e)}"
            )
        
        # Get STT service and check if it's ready
        stt_service = get_stt_service()
        if not stt_service:
            # Clean up file
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(
                status_code=503, 
                detail="Speech-to-text service not available"
            )
        
        if not stt_service.is_ready():
            # Clean up file
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(
                status_code=503,
                detail="Speech-to-text service not ready"
            )
        
        # Perform transcription
        try:
            transcription = await stt_service.transcribe_file(file_path)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            transcription = None
        finally:
            # Always clean up temporary file
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        
        if transcription is None:
            raise HTTPException(
                status_code=500, 
                detail="Transcription failed. Please ensure your audio file meets the requirements: WAV format, 16kHz sample rate, mono channel, 16-bit PCM"
            )
        
        # Get intent classification if available
        intent_service = get_intent_service()
        intent_result = None
        if intent_service and transcription:
            try:
                intent_result = await intent_service.classify_intent(transcription)
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}")
        
        # Get model info for response
        model_info = stt_service.get_model_info()
        
        return {
            "success": True,
            "transcription": transcription,
            "intent": intent_result,
            "user_id": current_user_id,
            "model_info": model_info,
            "audio_requirements": {
                "format": "WAV",
                "sample_rate": f"{settings.AUDIO_SAMPLE_RATE}Hz",
                "channels": "Mono",
                "bit_depth": f"{settings.AUDIO_BIT_DEPTH}-bit PCM"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during transcription"
        )

@router.get("/model-info")
async def get_model_info(current_user: dict = Depends(verify_firebase_token)):
    """Get information about the loaded STT model and requirements."""
    try:
        stt_service = get_stt_service()
        if not stt_service:
            raise HTTPException(
                status_code=503,
                detail="Speech-to-text service not available"
            )
        
        model_info = stt_service.get_model_info()
        
        return {
            "model_info": model_info,
            "audio_requirements": {
                "supported_formats": settings.SUPPORTED_AUDIO_FORMATS,
                "sample_rate": f"{settings.AUDIO_SAMPLE_RATE}Hz",
                "channels": settings.AUDIO_CHANNELS,
                "bit_depth": f"{settings.AUDIO_BIT_DEPTH}-bit",
                "max_file_size": f"{settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            },
            "preprocessing": {
                "automatic_resampling": True,
                "automatic_mono_conversion": True,
                "automatic_bit_depth_conversion": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get model info error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.post("/transcribe-and-respond")
async def transcribe_and_respond(
    audio_file: UploadFile = File(...),
    current_user: dict = Depends(verify_firebase_token)
):
    """Transcribe audio and generate AI response with optional TTS."""
    try:
        # First transcribe the audio
        transcribe_result = await transcribe_audio(audio_file, current_user)
        transcription = transcribe_result["transcription"]
        intent_result = transcribe_result["intent"]
        
        # Generate chat response
        chat_service = get_chat_service()
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not available")
        
        response_text = await chat_service.generate_response(
            transcription, 
            current_user["uid"], 
            context=intent_result
        )
        
        # Generate TTS audio for response
        tts_service = get_tts_service()
        audio_response = None
        if tts_service:
            audio_response = await tts_service.synthesize_speech(response_text)
        
        return {
            "transcription": transcription,
            "intent": intent_result,
            "response_text": response_text,
            "has_audio_response": audio_response is not None,
            "user_id": current_user["uid"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcribe and respond error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/response-audio/{text}")
async def get_response_audio(
    text: str,
    voice: Optional[str] = "default",
    current_user: dict = Depends(verify_firebase_token)
):
    """Generate TTS audio for given text."""
    try:
        tts_service = get_tts_service()
        if not tts_service:
            raise HTTPException(status_code=503, detail="TTS service not available")
        
        audio_data = await tts_service.synthesize_speech(text, voice)
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=response.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/voices")
async def get_available_voices(current_user: dict = Depends(verify_firebase_token)):
    """Get list of available TTS voices."""
    try:
        tts_service = get_tts_service()
        if not tts_service:
            raise HTTPException(status_code=503, detail="TTS service not available")
        
        voices = await tts_service.get_available_voices()
        return {"voices": voices}
        
    except Exception as e:
        logger.error(f"Get voices error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/intent-classify")
async def classify_intent(
    text: str,
    current_user: dict = Depends(verify_firebase_token)
):
    """Classify intent of given text."""
    try:
        intent_service = get_intent_service()
        if not intent_service:
            raise HTTPException(status_code=503, detail="Intent service not available")
        
        result = await intent_service.classify_intent(text)
        return result
        
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/intent-suggestions")
async def get_intent_suggestions(
    partial_text: str,
    current_user: dict = Depends(verify_firebase_token)
):
    """Get intent suggestions for partial text."""
    try:
        intent_service = get_intent_service()
        if not intent_service:
            raise HTTPException(status_code=503, detail="Intent service not available")
        
        suggestions = await intent_service.get_intent_suggestions(partial_text)
        return {"suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"Intent suggestions error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") 