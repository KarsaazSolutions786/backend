from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
import aiofiles
import os
import io
from core.security import get_current_user
from core.dependencies import get_stt_service, get_tts_service, get_intent_service, get_chat_service
from utils.logger import logger
from core.config import settings

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Transcribe uploaded audio file to text."""
    try:
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Check file size
        if audio_file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Save uploaded file temporarily
        file_path = os.path.join(settings.UPLOAD_DIR, f"temp_{current_user}_{audio_file.filename}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Get STT service and transcribe
        stt_service = get_stt_service()
        if not stt_service:
            raise HTTPException(status_code=503, detail="STT service not available")
        
        transcription = await stt_service.transcribe_file(file_path)
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        if transcription is None:
            raise HTTPException(status_code=500, detail="Transcription failed")
        
        # Get intent classification
        intent_service = get_intent_service()
        intent_result = None
        if intent_service:
            intent_result = await intent_service.classify_intent(transcription)
        
        return {
            "transcription": transcription,
            "intent": intent_result,
            "user_id": current_user
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/transcribe-and-respond")
async def transcribe_and_respond(
    audio_file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
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
            current_user, 
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
            "user_id": current_user
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
    current_user: str = Depends(get_current_user)
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
async def get_available_voices(current_user: str = Depends(get_current_user)):
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
    current_user: str = Depends(get_current_user)
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
    current_user: str = Depends(get_current_user)
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