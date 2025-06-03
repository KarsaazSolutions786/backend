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

def _generate_fallback_response(processing_result: dict) -> str:
    """Generate fallback response when Bloom 560M is not available."""
    intent = processing_result.get("intent", "unknown")
    data = processing_result.get("data", {})
    
    if intent == "create_reminder":
        response = f"Perfect! I've created a reminder titled '{data.get('title', 'your reminder')}'"
        if data.get("time"):
            response += f" scheduled for {data.get('time')}"
        response += ". I'll make sure to notify you when the time comes!"
        return response
        
    elif intent == "create_note":
        response = "Great! I've saved your note successfully. "
        if data.get("content"):
            response += "Your thoughts are now safely stored and you can access them anytime."
        else:
            response += "You can always come back to review or edit it later."
        return response
        
    elif intent in ["create_ledger", "add_expense"]:
        response = "Excellent! I've recorded your ledger entry"
        if data.get("amount") and data.get("person"):
            response += f" for ${data.get('amount')} with {data.get('person')}"
        elif data.get("amount"):
            response += f" for ${data.get('amount')}"
        response += ". Your financial records are now updated!"
        return response
        
    elif intent == "general_query":
        return "I understand your request and I'm here to help! Is there anything specific you'd like me to assist you with regarding reminders, notes, or your ledger?"
        
    else:
        return "I've processed your request successfully! Is there anything else you'd like me to help you with today?"

def _generate_multi_intent_fallback_response(processing_result: dict) -> str:
    """Generate fallback response for both single and multi-intent processing results."""
    
    # Check if this is a multi-intent result
    if "results" in processing_result and isinstance(processing_result.get("results"), list):
        return _generate_multi_intent_response(processing_result)
    else:
        # Single intent - use original function
        return _generate_fallback_response(processing_result)

def _generate_multi_intent_response(processing_result: dict) -> str:
    """Generate response for multi-intent processing results."""
    results = processing_result.get("results", [])
    total_intents = processing_result.get("total_intents", 0)
    successful_intents = processing_result.get("successful_intents", 0)
    
    if not results:
        return "I wasn't able to process your request. Please try again."
    
    if successful_intents == 0:
        return "I encountered some issues processing your requests. Please try again or be more specific."
    
    # Generate responses for each successful intent
    responses = []
    
    for result in results:
        if not result.get("success", False):
            continue
            
        intent = result.get("intent", "unknown")
        data = result.get("data", {})
        
        if intent == "create_reminder":
            if data.get("title"):
                response = f"✓ Created reminder: '{data.get('title')}'"
                if data.get("time"):
                    response += f" for {data.get('time')}"
            else:
                response = "✓ Created your reminder"
            responses.append(response)
            
        elif intent == "create_note":
            if data.get("content"):
                content_preview = data.get("content", "")[:30]
                if len(data.get("content", "")) > 30:
                    content_preview += "..."
                response = f"✓ Saved note: '{content_preview}'"
            else:
                response = "✓ Saved your note"
            responses.append(response)
            
        elif intent in ["create_ledger", "add_expense"]:
            if data.get("amount") and data.get("contact_name"):
                response = f"✓ Recorded ${data.get('amount')} with {data.get('contact_name')}"
            elif data.get("amount"):
                response = f"✓ Recorded ${data.get('amount')} in ledger"
            else:
                response = "✓ Added ledger entry"
            responses.append(response)
            
        elif intent in ["chit_chat", "general_query"]:
            response = "✓ Noted your message"
            responses.append(response)
    
    # Combine responses
    if len(responses) == 1:
        return f"Perfect! {responses[0]}. Is there anything else I can help you with?"
    elif len(responses) == 2:
        return f"Excellent! I've completed both tasks: {responses[0]} and {responses[1]}. Anything else I can do for you?"
    else:
        main_response = f"Great! I've completed {len(responses)} tasks: " + ", ".join(responses[:-1]) + f", and {responses[-1]}"
        return main_response + ". Is there anything else you need help with?"

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
            # Use multi-intent classification (returns {"intents": [...]} or single intent for backward compatibility)
            intent_result = await intent_service.classify_intent(transcription, multi_intent=True)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Intent classification failed: {str(e)}"
            )
        
        # Validate intent result
        has_intents = "intents" in intent_result and isinstance(intent_result.get("intents"), list) and len(intent_result.get("intents", [])) > 0
        has_single_intent = "intent" in intent_result and intent_result.get("intent")
        
        if not has_intents and not has_single_intent:
            raise HTTPException(
                status_code=500,
                detail="Intent classification returned invalid result"
            )
        
        processing_steps["intent_classification"] = True
        
        # Log intent results
        if has_intents:
            intents_info = [f"{intent['type']} ({intent.get('confidence', 0.0):.2f})" for intent in intent_result['intents']]
            logger.info(f"Step 3: Multi-intent classification completed: {len(intent_result['intents'])} intents - {', '.join(intents_info)}")
        else:
            logger.info(f"Step 3: Single intent classification completed: {intent_result['intent']} (confidence: {intent_result.get('confidence', 0.0)})")
        
        # === STEP 4: DATABASE PROCESSING ===
        logger.info("Step 4: Starting database processing")
        
        # Prepare intent data for processing (support both single and multi-intent)
        from services.intent_processor_service import IntentProcessorService
        intent_processor = IntentProcessorService()
        
        # The intent_result already contains the correct format for the processor
        # - For multi-intent: {"intents": [...], "original_text": "..."}
        # - For single-intent: {"intent": "...", "confidence": 0.95, "entities": {...}, "original_text": "..."}
        
        try:
            processing_result = await intent_processor.process_intent(
                intent_data=intent_result,
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
        
        # Log processing results
        if "results" in processing_result:
            # Multi-intent processing result
            total = processing_result.get("total_intents", 0)
            successful = processing_result.get("successful_intents", 0)
            logger.info(f"Step 4: Multi-intent database processing completed: {successful}/{total} intents processed successfully")
        else:
            # Single intent processing result
            logger.info(f"Step 4: Single intent database processing completed successfully for intent: {processing_result.get('intent')}")
        
        # === STEP 5: GENERATE AI RESPONSE WITH BLOOM 560M ===
        logger.info("Step 5: Generating AI response with Bloom 560M")
        
        # Get chat service for conversational AI response
        chat_service = get_chat_service()
        if chat_service and chat_service.is_ready():
            try:
                # Generate conversational response using Bloom 560M
                enhanced_response = await chat_service.generate_response(
                    message=transcription,  # Original user transcription
                    user_id=current_user_id,
                    context={
                        "intent_result": intent_result,
                        "processing_result": processing_result
                    }
                )
                
                if enhanced_response and enhanced_response.strip():
                    response_text = enhanced_response
                    logger.info(f"Bloom 560M generated response: {response_text[:100]}...")
                else:
                    logger.warning("Bloom 560M returned empty response, using fallback")
                    response_text = _generate_multi_intent_fallback_response(processing_result)
                    
            except Exception as e:
                logger.warning(f"Bloom 560M response generation failed, using fallback: {e}")
                response_text = _generate_multi_intent_fallback_response(processing_result)
        else:
            logger.warning("Chat service not available, using fallback response")
            response_text = _generate_multi_intent_fallback_response(processing_result)
        
        logger.info("Step 5: AI response generated successfully")
        
        # === STEP 6: GENERATE TTS AUDIO RESPONSE ===
        logger.info("Step 6: Generating TTS audio response")
        
        audio_response_url = None
        audio_response_data = None
        
        tts_service = get_tts_service()
        if tts_service and tts_service.is_ready():
            try:
                # Generate TTS audio for the AI response
                audio_data = await tts_service.synthesize_speech(response_text, voice="default")
                
                if audio_data:
                    # Save audio data to return in response
                    audio_response_data = audio_data
                    # Create a URL for audio streaming
                    audio_response_url = f"/api/v1/stt/response-audio/{response_text[:50]}..."
                    logger.info("TTS audio generated successfully")
                else:
                    logger.warning("TTS audio generation failed")
                    
            except Exception as e:
                logger.warning(f"TTS audio generation failed: {e}")
        else:
            logger.warning("TTS service not available")
        
        logger.info("Step 6: TTS processing completed")
        
        # === PREPARE FINAL RESPONSE ===
        final_response = {
            "success": True,
            "pipeline_completed": True,
            "processing_steps": processing_steps,
            "transcription": transcription,
            "intent_result": intent_result,
            "processing_result": processing_result,
            "ai_response": {
                "text": response_text,
                "audio_available": audio_response_data is not None,
                "audio_url": audio_response_url,
                "generated_by": "bloom_560m" if chat_service and chat_service.is_ready() else "fallback"
            },
            "user_id": current_user_id,
            "model_info": {
                "stt": stt_service.get_model_info() if stt_service else None,
                "chat": chat_service.get_model_info() if chat_service else None,
                "tts": tts_service.get_engine_info() if tts_service else None
            },
            "audio_requirements": {
                "format": "WAV",
                "sample_rate": f"{settings.AUDIO_SAMPLE_RATE}Hz",
                "channels": "Mono",
                "bit_depth": f"{settings.AUDIO_BIT_DEPTH}-bit PCM"
            }
        }
        
        # Include audio data if available (base64 encoded for JSON response)
        if audio_response_data:
            import base64
            final_response["ai_response"]["audio_base64"] = base64.b64encode(audio_response_data).decode('utf-8')
        
        logger.info(f"Complete voice-to-response pipeline completed successfully for user {current_user_id}")
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
    """Generate TTS audio for given text response."""
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
            headers={
                "Content-Disposition": f"attachment; filename=ai_response.wav",
                "Content-Length": str(len(audio_data))
            }
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

@router.post("/stream-conversation")
async def stream_conversation(
    audio_file: UploadFile = File(...),
    return_audio: bool = True,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Enhanced conversational endpoint: Audio in → Bloom 560M response → TTS Audio out
    
    This endpoint provides a complete voice-to-voice conversation experience:
    1. STT: Convert uploaded audio to text
    2. Bloom 560M: Generate conversational AI response
    3. TTS: Convert AI response to audio
    4. Return both text and audio responses
    """
    temp_file_path = None
    
    try:
        current_user_id = current_user["uid"]
        logger.info(f"Starting stream conversation for user: {current_user_id}")
        
        # === STEP 1: TRANSCRIBE AUDIO ===
        # Validate and save audio file
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {file_ext}")
        
        # Save temporary file
        temp_filename = f"conv_{current_user_id}_{audio_file.filename}"
        temp_file_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
        
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Transcribe audio
        stt_service = get_stt_service()
        if not stt_service:
            raise HTTPException(status_code=503, detail="STT service not available")
        
        transcription = await stt_service.transcribe_file(temp_file_path)
        if not transcription:
            raise HTTPException(status_code=500, detail="Transcription failed")
        
        logger.info(f"Transcribed: {transcription}")
        
        # === STEP 2: GENERATE AI RESPONSE WITH BLOOM 560M ===
        chat_service = get_chat_service()
        if not chat_service:
            raise HTTPException(status_code=503, detail="Chat service not available")
        
        # Generate conversational response
        ai_response = await chat_service.generate_response(
            message=transcription,
            user_id=current_user_id,
            context={"conversation_mode": True}
        )
        
        logger.info(f"AI Response: {ai_response}")
        
        # === STEP 3: GENERATE TTS AUDIO ===
        audio_data = None
        if return_audio:
            tts_service = get_tts_service()
            if tts_service:
                try:
                    audio_data = await tts_service.synthesize_speech(ai_response, voice)
                except Exception as e:
                    logger.warning(f"TTS generation failed: {e}")
        
        # === PREPARE RESPONSE ===
        response = {
            "success": True,
            "conversation": {
                "user_input": transcription,
                "ai_response": ai_response,
                "audio_available": audio_data is not None
            },
            "models_used": {
                "stt": stt_service.get_model_info() if stt_service else None,
                "chat": chat_service.get_model_info() if chat_service else None,
                "tts": tts_service.get_engine_info() if tts_service else None
            }
        }
        
        # Include audio data if available
        if audio_data:
            import base64
            response["conversation"]["audio_base64"] = base64.b64encode(audio_data).decode('utf-8')
            response["conversation"]["audio_url"] = f"/api/v1/stt/response-audio/{ai_response[:50]}"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream conversation error: {e}")
        raise HTTPException(status_code=500, detail=f"Conversation processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}") 