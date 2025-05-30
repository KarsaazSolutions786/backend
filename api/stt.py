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
    """
    Complete voice-to-database pipeline: transcribe audio, classify intent, and save to database.
    
    This endpoint performs the full pipeline:
    1. Validates and processes uploaded WAV audio file
    2. Uses Coqui STT to transcribe audio to text
    3. Classifies intent and extracts entities from transcription
    4. Processes and saves data to appropriate database table based on intent
    5. Returns the complete processing result
    
    Requirements:
    - File format: WAV (.wav)
    - Sample rate: 16 kHz
    - Channels: Mono (1 channel)
    - Bit depth: 16-bit PCM
    """
    temp_file_path = None
    processing_steps = {
        "audio_validation": False,
        "transcription": False,
        "intent_classification": False,
        "database_processing": False
    }
    
    try:
        # Get current user ID from Firebase token
        current_user_id = current_user["uid"]
        logger.info(f"Starting transcribe-and-respond pipeline for user: {current_user_id}")
        
        # === STEP 1: AUDIO FILE VALIDATION ===
        logger.info("Step 1: Validating audio file")
        
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
        temp_file_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
        
        # Save uploaded file temporarily
        try:
            async with aiofiles.open(temp_file_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded file: {str(e)}"
            )
        
        processing_steps["audio_validation"] = True
        logger.info("Step 1: Audio validation completed successfully")
        
        # === STEP 2: SPEECH-TO-TEXT TRANSCRIPTION ===
        logger.info("Step 2: Starting STT transcription")
        
        # Get STT service and check if it's ready
        stt_service = get_stt_service()
        if not stt_service:
            raise HTTPException(
                status_code=503, 
                detail="Speech-to-text service not available"
            )
        
        if not stt_service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Speech-to-text service not ready"
            )
        
        # Perform transcription
        try:
            transcription = await stt_service.transcribe_file(temp_file_path)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            transcription = None
        
        if transcription is None or transcription.strip() == "":
            raise HTTPException(
                status_code=500, 
                detail="Transcription failed or returned empty result. Please ensure your audio file meets the requirements: WAV format, 16kHz sample rate, mono channel, 16-bit PCM, and contains clear speech"
            )
        
        processing_steps["transcription"] = True
        logger.info(f"Step 2: Transcription completed successfully: '{transcription}'")
        
        # === STEP 3: INTENT CLASSIFICATION ===
        logger.info("Step 3: Starting intent classification")
        
        intent_service = get_intent_service()
        if not intent_service:
            raise HTTPException(
                status_code=503,
                detail="Intent classification service not available"
            )
        
        try:
            intent_result = await intent_service.classify_intent(transcription)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Intent classification failed: {str(e)}"
            )
        
        if not intent_result or not intent_result.get("intent"):
            raise HTTPException(
                status_code=500,
                detail="Intent classification returned invalid result"
            )
        
        processing_steps["intent_classification"] = True
        logger.info(f"Step 3: Intent classification completed: {intent_result['intent']} (confidence: {intent_result.get('confidence', 0.0)})")
        
        # === STEP 4: DATABASE PROCESSING ===
        logger.info("Step 4: Starting database processing")
        
        # Prepare intent data for processing
        from services.intent_processor_service import IntentProcessorService
        intent_processor = IntentProcessorService()
        
        intent_data = {
            "intent": intent_result.get("intent"),
            "confidence": intent_result.get("confidence", 0.0),
            "entities": intent_result.get("entities", {}),
            "original_text": transcription
        }
        
        try:
            processing_result = await intent_processor.process_intent(
                intent_data=intent_data,
                user_id=current_user_id
            )
        except Exception as e:
            logger.error(f"Database processing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Database processing failed: {str(e)}"
            )
        
        if not processing_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Database processing failed: {processing_result.get('error', 'Unknown error')}"
            )
        
        processing_steps["database_processing"] = True
        logger.info(f"Step 4: Database processing completed successfully for intent: {processing_result.get('intent')}")
        
        # === STEP 5: GENERATE AI RESPONSE (Optional) ===
        logger.info("Step 5: Generating AI response")
        
        # Generate contextual response based on processing result
        response_text = ""
        if processing_result.get("intent") == "create_reminder":
            data = processing_result.get("data", {})
            response_text = f"I've created a reminder titled '{data.get('title', 'Reminder')}'"
            if data.get("time"):
                response_text += f" for {data.get('time')}"
            response_text += "."
        elif processing_result.get("intent") == "create_note":
            response_text = "I've saved your note successfully."
        elif processing_result.get("intent") in ["create_ledger", "add_expense"]:
            data = processing_result.get("data", {})
            response_text = f"I've recorded the ledger entry"
            if data.get("amount") and data.get("person"):
                response_text += f" for ${data.get('amount')} with {data.get('person')}"
            response_text += "."
        else:
            response_text = "I've processed your request successfully."
        
        # Try to get chat service for enhanced response
        chat_service = get_chat_service()
        if chat_service:
            try:
                enhanced_response = await chat_service.generate_response(
                    transcription, 
                    current_user_id, 
                    context=intent_result
                )
                if enhanced_response and enhanced_response.strip():
                    response_text = enhanced_response
            except Exception as e:
                logger.warning(f"Enhanced response generation failed, using default: {e}")
        
        logger.info("Step 5: AI response generated successfully")
        
        # === PREPARE FINAL RESPONSE ===
        final_response = {
            "success": True,
            "pipeline_completed": True,
            "processing_steps": processing_steps,
            "transcription": transcription,
            "intent_result": intent_result,
            "processing_result": processing_result,
            "response_text": response_text,
            "user_id": current_user_id,
            "model_info": stt_service.get_model_info() if stt_service else None,
            "audio_requirements": {
                "format": "WAV",
                "sample_rate": f"{settings.AUDIO_SAMPLE_RATE}Hz",
                "channels": "Mono",
                "bit_depth": f"{settings.AUDIO_BIT_DEPTH}-bit PCM"
            }
        }
        
        logger.info(f"Pipeline completed successfully for user {current_user_id}")
        return final_response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in transcribe-and-respond pipeline: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during voice processing pipeline: {str(e)}"
        )
    finally:
        # Always clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")

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