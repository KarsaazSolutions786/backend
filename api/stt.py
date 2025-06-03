from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, Dict, Any
import aiofiles
import os
import io
from pathlib import Path
from firebase_auth import verify_firebase_token
from core.dependencies import get_stt_service, get_tts_service, get_intent_service, get_chat_service, get_intent_processor_service
from utils.logger import logger
from core.config import settings

# Placeholder for a proper i18n/l10n library (same as in IntentProcessorService)
def get_localized_string(key: str, lang: str, **kwargs) -> str:
    translations_en = {
        "reminder_created_successfully": "Perfect! I've created a reminder titled '{title}'",
        "reminder_scheduled_successfully": "Perfect! I've created a reminder titled '{title}' scheduled for {time}. I'll make sure to notify you!",
        "note_created_successfully": "Great! I've saved your note successfully. Your thoughts are now safely stored.",
        "note_created_with_content_successfully": "Great! I've saved your note: '{preview}'. Your thoughts are now safely stored.",
        "ledger_entry_created_successfully": "Excellent! I've recorded your ledger entry. Your financial records are now updated!",
        "ledger_entry_created_with_details_successfully": "Excellent! I've recorded ${amount} with {person}. Your financial records are now updated!",
        "ledger_entry_created_with_amount_successfully": "Excellent! I've recorded ${amount} in your ledger. Your financial records are now updated!",
        "general_query_response": "I understand your request and I'm here to help! Is there anything specific you'd like me to assist you with regarding reminders, notes, or your ledger?",
        "processed_successfully_default": "I've processed your request successfully! Is there anything else you'd like me to help you with today?",
        "multi_intent_response_one": "Perfect! {response1}. Is there anything else I can help you with?",
        "multi_intent_response_two": "Excellent! I've completed both tasks: {response1} and {response2}. Anything else I can do for you?",
        "multi_intent_response_many": "Great! I've completed {count} tasks: {responses_list_str}, and {last_response}. Is there anything else you need help with?",
        "multi_intent_no_results": "I wasn't able to process your request. Please try again.",
        "multi_intent_all_failed": "I encountered some issues processing your requests. Please try again or be more specific.",
        "multi_reminder_created": "✓ Created reminder: '{title}'",
        "multi_reminder_created_with_time": "✓ Created reminder: '{title}' for {time}",
        "multi_note_saved_preview": "✓ Saved note: '{preview}'",
        "multi_note_saved": "✓ Saved your note",
        "multi_ledger_recorded_details": "✓ Recorded ${amount} with {contact_name}",
        "multi_ledger_recorded_amount": "✓ Recorded ${amount} in ledger",
        "multi_ledger_recorded": "✓ Added ledger entry",
        "multi_chat_logged": "✓ Noted your message",
        "error_stt_not_available": "Speech-to-text service not available",
        "error_stt_not_ready": "Speech-to-text service not ready",
        "error_transcription_failed_requirements": "Transcription failed. Please ensure your audio file meets the requirements: WAV format, 16kHz sample rate, mono channel, 16-bit PCM",
        "error_intent_classification_failed": "Intent classification failed: {error}",
        "error_chat_service_not_available": "Chat service is not available or ready to generate a response.",
        "error_processing_failed": "Processing failed: {error}",
        "error_tts_failed": "Text-to-speech conversion failed: {error}",
        "error_stt_failed_specific": "Speech-to-text processing failed: {error}"
    }
    if lang == "en":
        return translations_en.get(key, key).format(**kwargs)
    # Add other languages here
    # elif lang == "es":
    #     return translations_es.get(key, key).format(**kwargs)
    return key.format(**kwargs) # Fallback

router = APIRouter()

def _generate_fallback_response(processing_result: dict, language_code: str = "en") -> str:
    intent = processing_result.get("intent", "unknown")
    data = processing_result.get("data", {})
    
    if intent == "create_reminder":
        title = data.get('title', 'your reminder') # Assuming title is already generated considering language
        time = data.get("time")
        if time:
            return get_localized_string("reminder_scheduled_successfully", language_code, title=title, time=time)
        return get_localized_string("reminder_created_successfully", language_code, title=title)
        
    elif intent == "create_note":
        content = data.get("content")
        if content:
            preview = content[:30] + "..." if len(content) > 30 else content
            return get_localized_string("note_created_with_content_successfully", language_code, preview=preview)
        return get_localized_string("note_created_successfully", language_code)
        
    elif intent in ["create_ledger", "add_expense"]:
        amount = data.get("amount")
        person = data.get("person") # or contact_name from processor
        if amount and person:
            return get_localized_string("ledger_entry_created_with_details_successfully", language_code, amount=amount, person=person)
        elif amount:
            return get_localized_string("ledger_entry_created_with_amount_successfully", language_code, amount=amount)
        return get_localized_string("ledger_entry_created_successfully", language_code)
        
    elif intent == "general_query":
        return get_localized_string("general_query_response", language_code)
        
    else:
        return get_localized_string("processed_successfully_default", language_code)

def _generate_multi_intent_fallback_response(processing_result: dict, language_code: str = "en") -> str:
    if "results" in processing_result and isinstance(processing_result.get("results"), list):
        return _generate_multi_intent_response(processing_result, language_code)
    else:
        return _generate_fallback_response(processing_result, language_code)

def _generate_multi_intent_response(processing_result: dict, language_code: str = "en") -> str:
    results = processing_result.get("results", [])
    successful_intents = processing_result.get("successful_intents", 0)
    
    if not results: return get_localized_string("multi_intent_no_results", language_code)
    if successful_intents == 0: return get_localized_string("multi_intent_all_failed", language_code)
    
    responses = []
    for result in results:
        if not result.get("success", False): continue
        intent = result.get("intent", "unknown"); data = result.get("data", {})
        
        if intent == "create_reminder":
            title = data.get('title', 'your reminder')
            time = data.get("time")
            key = "multi_reminder_created_with_time" if time else "multi_reminder_created"
            responses.append(get_localized_string(key, language_code, title=title, time=time))
        elif intent == "create_note":
            content = data.get("content", "")
            if content:
                preview = content[:30] + "..." if len(content) > 30 else content
                responses.append(get_localized_string("multi_note_saved_preview", language_code, preview=preview))
            else:
                responses.append(get_localized_string("multi_note_saved", language_code))
        elif intent in ["create_ledger", "add_expense"]:
            amount = data.get("amount"); contact_name = data.get("contact_name")
            if amount and contact_name:
                responses.append(get_localized_string("multi_ledger_recorded_details", language_code, amount=amount, contact_name=contact_name))
            elif amount:
                responses.append(get_localized_string("multi_ledger_recorded_amount", language_code, amount=amount))
            else: responses.append(get_localized_string("multi_ledger_recorded", language_code))
        elif intent in ["chit_chat", "general_query"]:
            responses.append(get_localized_string("multi_chat_logged", language_code))
    
    if not responses: return get_localized_string("multi_intent_all_failed", language_code) # if all successful intents had no response string
    if len(responses) == 1: return get_localized_string("multi_intent_response_one", language_code, response1=responses[0])
    if len(responses) == 2: return get_localized_string("multi_intent_response_two", language_code, response1=responses[0], response2=responses[1])
    
    responses_list_str = ", ".join(responses[:-1])
    return get_localized_string("multi_intent_response_many", language_code, count=len(responses), responses_list_str=responses_list_str, last_response=responses[-1])

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
    language_code: Optional[str] = Query("en", description="ISO 639-1 language code for STT, Intent, and TTS processing"),
    current_user: dict = Depends(verify_firebase_token)
):
    current_user_id = current_user["uid"]
    logger.info(f"User {current_user_id} called /transcribe-and-respond with language: {language_code}")

    # File validation (remains the same)
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format. Supported: {settings.SUPPORTED_AUDIO_FORMATS}")
    if audio_file.size and audio_file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large.")

    temp_filename = f"temp_{current_user_id}_{Path(audio_file.filename).name}"
    file_path = settings.UPLOAD_DIR / temp_filename
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(await audio_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    transcription = None
    intent_result = None
    processing_result = None
    response_text = None
    final_response_audio_url = None

    try:
        stt_service = get_stt_service() # STT service might need lang later
        if not stt_service or not stt_service.is_ready():
            raise HTTPException(status_code=503, detail=get_localized_string("error_stt_not_available", language_code))
        
        # STT service should eventually take language_code
        try:
            transcription = await stt_service.transcribe_file(str(file_path), language_code=language_code)
            if transcription is None: # If transcribe_file can return None on non-exception failure
                logger.warning(f"STT service returned None for transcription with lang {language_code}. File: {file_path}")
                # Use a more general transcription failure message if STT returns None without an exception
                raise HTTPException(status_code=500, detail=get_localized_string("error_transcription_failed_requirements", language_code))
        except HTTPException: # Re-raise HTTPExceptions if stt_service itself raises one (e.g., from internal validation)
            raise
        except Exception as e:
            logger.error(f"STT service failed during transcription for lang {language_code}: {e}", exc_info=True)
            # Use the new specific error key
            raise HTTPException(status_code=500, detail=get_localized_string("error_stt_failed_specific", language_code, error=str(e)))
        
        logger.info(f"Transcription ('{language_code}'): {transcription}")

        # Intent Service - now language aware
        intent_service = None
        try:
            intent_service = get_intent_service(language_code=language_code)
        except Exception as e:
            logger.error(f"Failed to get intent service for language {language_code}: {e}", exc_info=True)
            # Continue without intent service - we'll handle this below
        
        if not intent_service or not intent_service.is_ready():
            logger.warning(f"Intent service not available/ready for lang {language_code}. Proceeding without intent classification.")
        else:
            try:
                intent_result = await intent_service.classify_intent(transcription)
                logger.info(f"Intent result ('{language_code}'): {intent_result}")
            except Exception as e:
                logger.error(f"Intent classification failed for lang {language_code}: {e}", exc_info=True)
                # Not raising HTTPException here, will try to respond with just transcription
                intent_result = None
        
        # Intent Processing Service - now language aware
        if intent_result:
            try:
                intent_processor_service = get_intent_processor_service() # This is a singleton, language is passed to process_intent
                if not intent_processor_service:
                    logger.warning(f"Intent processor service not available for lang {language_code}")
                    processing_result = None
                else:
                    try:
                        processing_result = await intent_processor_service.process_intent(intent_result, current_user_id, language_code)
                        logger.info(f"Processing result ('{language_code}'): {processing_result}")
                    except Exception as e:
                        logger.error(f"Intent processing failed for lang {language_code}: {e}", exc_info=True)
                        # Continue without processing result - we'll generate a fallback response
                        processing_result = None
            except Exception as e:
                logger.error(f"Failed to get intent processor service for lang {language_code}: {e}", exc_info=True)
                processing_result = None
        else:
            processing_result = None
        
        # Response Generation
        # Use chat service if intent is general_query or no specific intent was processed successfully
        # This part needs careful thought on when to use chat vs. fallback for multilingual.
        should_use_chat = False
        if processing_result and "results" in processing_result: # Multi-intent
            successful_actions = sum(1 for r in processing_result["results"] if r.get("success") and r.get("intent") not in ["general_query", "chit_chat"])
            if successful_actions == 0: should_use_chat = True
        elif processing_result and processing_result.get("intent") in ["general_query", "chit_chat"]:
            should_use_chat = True
        elif not processing_result and intent_result and intent_result.get("intent") in ["general_query", "chit_chat"]:
            should_use_chat = True
        elif not processing_result and not intent_result: # Only transcription available
             should_use_chat = True # Or a specific message like "I have your transcription, what next?"

        if should_use_chat and settings.ENABLE_CHAT_SERVICE:
            try:
                chat_service = get_chat_service() # Chat service might also need language_code
                if not chat_service:
                    logger.warning(f"Chat service not available for lang {language_code}. Using fallback response.")
                    response_text = _generate_multi_intent_fallback_response(processing_result if processing_result else {}, language_code)
                elif not chat_service.is_ready():
                    logger.warning(f"Chat service not ready for lang {language_code}. Using fallback response.")
                    response_text = _generate_multi_intent_fallback_response(processing_result if processing_result else {}, language_code)
                else:
                    try:
                        # Chat service should ideally take language_code for its own NLU/NLG if any
                        response_text = await chat_service.generate_response(transcription, current_user_id, language_code=language_code)
                        logger.info(f"Chat service response ('{language_code}'): {response_text}")
                    except Exception as e:
                        logger.error(f"Chat service failed for lang {language_code}: {e}", exc_info=True)
                        response_text = _generate_multi_intent_fallback_response(processing_result if processing_result else {}, language_code)
            except Exception as e:
                logger.error(f"Failed to get chat service for lang {language_code}: {e}", exc_info=True)
                response_text = _generate_multi_intent_fallback_response(processing_result if processing_result else {}, language_code)
        elif processing_result: # Use fallback based on processed intents
            response_text = _generate_multi_intent_fallback_response(processing_result, language_code)
        else: # Only transcription available, or intent classification failed but no processing occurred
            response_text = get_localized_string("processed_successfully_default", language_code) # Generic fallback

        # TTS Service for audio response
        if settings.ENABLE_TTS_SERVICE and response_text:
            try:
                tts_service = get_tts_service() # TTS service needs language_code & voice selection based on language
                if not tts_service:
                    logger.warning(f"TTS service not available for lang {language_code}.")
                elif not tts_service.is_ready():
                    logger.warning(f"TTS service not ready for lang {language_code}.")
                else:
                    try:
                        # This is a simplified call. In reality, you'd select a voice compatible with language_code.
                        audio_content_bytes = await tts_service.convert_text_to_speech(response_text, language_code=language_code)
                        # Save to a publicly accessible URL or stream directly (more complex with FastAPI)
                        # For demo, assume a function to save and get URL:
                        if audio_content_bytes: # Ensure there is content to save
                             final_response_audio_url = await tts_service.save_speech_to_public_url(audio_content_bytes, current_user_id, language_code)
                        else:
                            logger.warning(f"TTS service returned no audio content for lang {language_code}. Text: '{response_text[:50]}...'")
                    except Exception as e:
                        logger.error(f"TTS service failed for lang {language_code}: {e}", exc_info=True)
                        # Proceed with text response only, do not raise an HTTPException that would become a 500 error for this part.
                        # Instead, the client will see that response_audio_url is null.
                        # If we wanted to return an error specifically for TTS failure while other parts succeeded, that would be a different design.
            except Exception as e:
                logger.error(f"Failed to get TTS service for lang {language_code}: {e}", exc_info=True)
                # Continue without TTS

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Unhandled error in /transcribe-and-respond for lang {language_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=get_localized_string("error_processing_failed", language_code, error=str(e)))
    finally:
        if Path(file_path).exists():
            try: os.remove(file_path)
            except Exception as e: logger.warning(f"Failed to remove temp file {file_path}: {e}")

    return {
        "success": True,
        "transcription": transcription,
        "intent_result": intent_result, # Original intent classification result
        "processing_result": processing_result, # Result from IntentProcessorService
        "response_text": response_text,
        "response_audio_url": final_response_audio_url,
        "language_code": language_code,
        "user_id": current_user_id
    }

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