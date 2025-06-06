"""
AI Pipeline API Endpoints
Provides endpoints for the complete AI pipeline workflow using Whisper STT, MiniLM Intent Classification, and Coqui TTS.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, Dict, Any
import aiofiles
import os
import io
from pathlib import Path
from firebase_auth import verify_firebase_token
from services.ai_pipeline_service import AIPipelineService
from utils.logger import logger

router = APIRouter()

# Initialize the AI pipeline service
ai_pipeline = AIPipelineService()

@router.post("/complete-pipeline")
async def complete_ai_pipeline(
    audio_file: UploadFile = File(...),
    language: str = "en",
    multi_intent: bool = True,
    store_in_database: bool = True,
    generate_audio_response: bool = True,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Complete AI Pipeline: Audio → Whisper STT → MiniLM Intent → Database → Coqui TTS
    
    This endpoint processes audio through the entire AI pipeline:
    1. Validates and preprocesses audio file
    2. Transcribes speech using Whisper STT
    3. Classifies intent(s) using MiniLM
    4. Stores results in database
    5. Generates TTS response using Coqui TTS
    
    Args:
        audio_file: WAV audio file (16kHz, mono, 16-bit PCM recommended)
        language: Language code for transcription (default: "en")
        multi_intent: Whether to detect multiple intents (default: True)
        store_in_database: Whether to save results to database (default: True)
        generate_audio_response: Whether to generate TTS response (default: True)
        voice: Voice ID for TTS response (default: "default")
        current_user: Authenticated user from Firebase
        
    Returns:
        Complete pipeline results including transcription, intent classification, database storage, and TTS response
    """
    try:
        # Get user ID
        user_id = current_user["uid"]
        
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Check file extension
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in ['.wav', '.mp3', '.m4a', '.flac']:
            raise HTTPException(
                status_code=400,
                detail="Supported audio formats: WAV, MP3, M4A, FLAC"
            )
        
        # Save uploaded file temporarily
        temp_path = f"uploads/{audio_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
            
            logger.info(f"Processing complete AI pipeline for user {user_id}, file: {audio_file.filename}")
            
            # Process through complete AI pipeline
            result = await ai_pipeline.process_complete_pipeline(
                audio_file_path=temp_path,
                user_id=user_id,
                language=language,
                multi_intent=multi_intent,
                store_in_database=store_in_database,
                generate_audio_response=generate_audio_response,
                voice=voice
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if not result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=f"Pipeline processing failed: {'; '.join(result.get('errors', ['Unknown error']))}"
                )
            
            # Prepare response (exclude large audio data from main response)
            response_data = {
                "success": True,
                "pipeline_completed": result.get("pipeline_completed", False),
                "processing_steps": result.get("processing_steps", {}),
                "user_id": user_id,
                "processing_time": result.get("processing_time", 0.0),
                "transcription": result.get("transcription", ""),
                "transcription_confidence": result.get("transcription_confidence", 0.0),
                "language_detected": result.get("language_detected", language),
                "intent_result": result.get("intent_result", {}),
                "database_result": result.get("database_result", {}),
                "tts_result": {
                    "success": result.get("tts_result", {}).get("success", False),
                    "response_text": result.get("tts_result", {}).get("response_text", ""),
                    "audio_size": result.get("tts_result", {}).get("audio_size", 0),
                    "voice_used": result.get("tts_result", {}).get("voice_used", voice),
                    "audio_available": "audio_data" in result.get("tts_result", {})
                },
                "errors": result.get("errors", [])
            }
            
            logger.info(f"AI pipeline completed successfully for user {user_id} in {result.get('processing_time', 0):.2f}s")
            return response_data
            
        except Exception as e:
            # Clean up temporary file on error
            try:
                os.remove(temp_path)
            except:
                pass
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete AI pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing failed: {str(e)}"
        )

@router.get("/pipeline-audio-response/{user_id}")
async def get_pipeline_audio_response(
    user_id: str,
    text: str,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
) -> StreamingResponse:
    """
    Generate audio response using Coqui TTS.
    
    Args:
        user_id: User ID (must match authenticated user)
        text: Text to convert to speech
        voice: Voice ID to use
        current_user: Authenticated user
        
    Returns:
        Audio stream (WAV format)
    """
    try:
        # Verify user authorization
        if current_user["uid"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied: User ID mismatch"
            )
        
        # Generate audio
        audio_data = await ai_pipeline.generate_speech_audio(text, voice)
        
        if not audio_data:
            raise HTTPException(
                status_code=500,
                detail="TTS generation failed"
            )
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=response_{user_id}.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio response generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}"
        )

@router.post("/transcribe-only")
async def transcribe_audio_only(
    audio_file: UploadFile = File(...),
    language: str = "en",
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper STT only (no intent classification or database storage).
    
    Args:
        audio_file: Audio file to transcribe
        language: Language code for transcription
        current_user: Authenticated user
        
    Returns:
        Transcription results
    """
    try:
        user_id = current_user["uid"]
        
        # Validate file
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Save file temporarily
        temp_path = f"uploads/{audio_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
            
            # Transcribe only
            result = await ai_pipeline.transcribe_audio_only(temp_path, language)
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Add user info
            result["user_id"] = user_id
            
            return result
            
        except Exception as e:
            try:
                os.remove(temp_path)
            except:
                pass
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@router.post("/classify-text-intent")
async def classify_text_intent(
    text: str,
    multi_intent: bool = True,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Classify intent from text using MiniLM (no audio processing).
    
    Args:
        text: Text to classify
        multi_intent: Whether to detect multiple intents
        current_user: Authenticated user
        
    Returns:
        Intent classification results
    """
    try:
        user_id = current_user["uid"]
        
        # Classify intent
        result = await ai_pipeline.classify_text_intent(text, multi_intent)
        
        # Add user info
        result["user_id"] = user_id
        
        return result
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Intent classification failed: {str(e)}"
        )

@router.post("/generate-speech")
async def generate_speech_from_text(
    text: str,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
) -> StreamingResponse:
    """
    Generate speech audio from text using Coqui TTS.
    
    Args:
        text: Text to convert to speech
        voice: Voice ID to use
        current_user: Authenticated user
        
    Returns:
        Audio stream (WAV format)
    """
    try:
        user_id = current_user["uid"]
        
        # Generate audio
        audio_data = await ai_pipeline.generate_speech_audio(text, voice)
        
        if not audio_data:
            raise HTTPException(
                status_code=500,
                detail="TTS generation failed"
            )
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=tts_{user_id}.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {str(e)}"
        )

@router.get("/service-status")
async def get_ai_service_status(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get status of all AI services in the pipeline.
    
    Returns:
        Status information for all AI services
    """
    try:
        status = ai_pipeline.get_service_status()
        status["user_id"] = current_user["uid"]
        status["pipeline_ready"] = ai_pipeline.is_ready()
        
        return status
        
    except Exception as e:
        logger.error(f"Service status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Service status check failed: {str(e)}"
        )

@router.get("/available-voices")
async def get_available_voices(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get list of available TTS voices.
    
    Returns:
        List of available voices for TTS
    """
    try:
        voices = await ai_pipeline.coqui_tts.get_available_voices()
        
        return {
            "success": True,
            "voices": voices,
            "user_id": current_user["uid"]
        }
        
    except Exception as e:
        logger.error(f"Voice list retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice list retrieval failed: {str(e)}"
        )

@router.get("/model-info")
async def get_model_information(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get detailed information about all loaded AI models.
    
    Returns:
        Detailed model information
    """
    try:
        return {
            "success": True,
            "models": {
                "whisper_stt": ai_pipeline.whisper_stt.get_model_info(),
                "minilm_intent": ai_pipeline.minilm_intent.get_model_info(),
                "tts_service": ai_pipeline.coqui_tts.get_engine_info()
            },
            "pipeline_ready": ai_pipeline.is_ready(),
            "user_id": current_user["uid"]
        }
        
    except Exception as e:
        logger.error(f"Model info retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model info retrieval failed: {str(e)}"
        ) 