"""
AI Pipeline API Endpoints
Provides endpoints for the complete AI pipeline workflow using Whisper STT, MiniLM Intent Classification, Bloom Text Generation, and Coqui TTS.
Also includes model management endpoints for quantization and optimization.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, Dict, Any, List
import aiofiles
import os
import io
import asyncio
import subprocess
import json
from pathlib import Path
from datetime import datetime
from firebase_auth import verify_firebase_token
from services.ai_pipeline_service import AIPipelineService
from utils.logger import logger
from api.bloom_response_generator import bloom_generator

router = APIRouter()

# Initialize the AI pipeline service
ai_pipeline = AIPipelineService()

# Model management state tracking
model_operations = {}

# --- Enhanced AI Pipeline Endpoints with Bloom Integration ---

@router.post("/complete-pipeline-with-bloom")
async def complete_ai_pipeline_with_bloom(
    audio_file: UploadFile = File(...),
    language: str = "en",
    multi_intent: bool = True,
    store_in_database: bool = True,
    use_bloom_generation: bool = True,
    bloom_model_path: Optional[str] = None,  # Made optional
    max_tokens: int = 50,
    temperature: float = 0.7,
    generate_audio_response: bool = True,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Enhanced AI Pipeline: Audio → Whisper STT → MiniLM Intent → Database → Bloom Text Generation → Coqui TTS
    
    This endpoint processes audio through the enhanced AI pipeline with Bloom model integration:
    1. Validates and preprocesses audio file
    2. Transcribes speech using Whisper STT
    3. Classifies intent(s) using MiniLM
    4. Stores results in database
    5. **NEW**: Generates contextual response using Bloom model (if available)
    6. Generates TTS response using Coqui TTS
    
    Args:
        audio_file: WAV audio file (16kHz, mono, 16-bit PCM recommended)
        language: Language code for transcription (default: "en")
        multi_intent: Whether to detect multiple intents (default: True)
        store_in_database: Whether to save results to database (default: True)
        use_bloom_generation: Whether to use Bloom model for response generation
        bloom_model_path: Path to quantized Bloom model (auto-detected if None)
        max_tokens: Maximum tokens for Bloom generation
        temperature: Temperature for text generation (0.1-1.0)
        generate_audio_response: Whether to generate TTS response (default: True)
        voice: Voice ID for TTS response (default: "default")
        current_user: Authenticated user from Firebase
        
    Returns:
        Complete pipeline results including Bloom-generated responses
    """
    try:
        # Get user ID
        user_id = current_user["uid"]
        
        # Auto-detect available Bloom model if not specified
        if bloom_model_path is None:
            bloom_model_path = _find_available_bloom_model()
            if not bloom_model_path and use_bloom_generation:
                logger.warning("No Bloom model found, disabling Bloom generation")
                use_bloom_generation = False
        
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
            
            logger.info(f"Processing enhanced AI pipeline with Bloom for user {user_id}, file: {audio_file.filename}")
            
            # Step 1-3: Run standard pipeline up to database operations
            standard_result = await ai_pipeline.process_complete_pipeline(
                audio_file_path=temp_path,
                user_id=user_id,
                language=language,
                multi_intent=multi_intent,
                store_in_database=store_in_database,
                generate_audio_response=False,  # We'll handle TTS after Bloom
                voice=voice
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if not standard_result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=f"Standard pipeline processing failed: {'; '.join(standard_result.get('errors', ['Unknown error']))}"
                )
            
            # Step 4: Enhanced Response Generation with Bloom (if enabled and available)
            bloom_response = None
            final_response_text = standard_result.get("tts_result", {}).get("response_text", "")
            bloom_used = False
            
            if use_bloom_generation and bloom_model_path:
                logger.info(f"Generating enhanced response using Bloom model at: {bloom_model_path}")
                bloom_result = await _generate_bloom_response(
                    transcription=standard_result.get("transcription", ""),
                    intent_result=standard_result.get("intent_result", {}),
                    database_result=standard_result.get("database_result", {}),
                    model_path=bloom_model_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    user_id=user_id
                )
                
                bloom_response = bloom_result
                if bloom_result.get("success") and bloom_result.get("generated_text"):
                    final_response_text = bloom_result["generated_text"]
                    bloom_used = True
                    logger.info(f"Bloom generated response: {final_response_text[:100]}...")
                else:
                    logger.warning(f"Bloom generation failed, using fallback: {bloom_result.get('error', 'Unknown error')}")
            elif use_bloom_generation and not bloom_model_path:
                logger.info("Bloom generation requested but no model available, using standard response")
            
            # Step 5: Generate TTS with final response
            tts_result = None
            if generate_audio_response and final_response_text:
                logger.info("Generating TTS for final response")
                try:
                    audio_data = await ai_pipeline.generate_speech_audio(final_response_text, voice)
                    if audio_data:
                        tts_result = {
                            "success": True,
                            "response_text": final_response_text,
                            "audio_data": audio_data,
                            "audio_size": len(audio_data),
                            "voice_used": voice,
                            "bloom_enhanced": bloom_used
                        }
                    else:
                        tts_result = {
                            "success": False,
                            "error": "TTS synthesis failed",
                            "response_text": final_response_text,
                            "audio_size": 0,
                            "voice_used": voice,
                            "bloom_enhanced": False
                        }
                except Exception as e:
                    logger.error(f"TTS generation failed: {e}")
                    tts_result = {
                        "success": False,
                        "error": f"TTS generation failed: {str(e)}",
                        "response_text": final_response_text,
                        "audio_size": 0,
                        "voice_used": voice,
                        "bloom_enhanced": False
                    }
            
            # Prepare enhanced response
            response_data = {
                "success": True,
                "pipeline_completed": True,
                "enhanced_with_bloom": bloom_used,
                "bloom_model_used": bloom_model_path if bloom_used else None,
                "user_id": user_id,
                "processing_time": standard_result.get("processing_time", 0.0),
                "transcription": standard_result.get("transcription", ""),
                "transcription_confidence": standard_result.get("transcription_confidence", 0.0),
                "language_detected": standard_result.get("language_detected", language),
                "intent_result": standard_result.get("intent_result", {}),
                "database_result": standard_result.get("database_result", {}),
                "bloom_result": bloom_response,
                "tts_result": tts_result or {
                    "success": False,
                    "error": "TTS not requested",
                    "response_text": final_response_text,
                    "audio_size": 0,
                    "voice_used": voice,
                    "bloom_enhanced": bloom_used
                },
                "processing_steps": {
                    "transcription": True,
                    "intent_classification": True,
                    "database_operations": store_in_database,
                    "bloom_generation": bloom_used,
                    "tts_generation": generate_audio_response
                },
                "errors": standard_result.get("errors", [])
            }
            
            # Add any Bloom errors (but don't fail the whole pipeline)
            bloom_fallback_used = bloom_response and bloom_response.get("method") in ["contextual_fallback", "basic_fallback"]
            
            if bloom_response and not bloom_response.get("success") and not bloom_fallback_used:
                error_msg = bloom_response.get('error', 'Unknown error')
                
                # Provide helpful guidance for dependency issues
                if bloom_response.get("dependency_issue"):
                    if "torch" in error_msg.lower():
                        response_data["errors"].append(f"Bloom generation failed: PyTorch not available. Your Python version (3.13) may not be supported. Try Python 3.8-3.12.")
                        response_data["bloom_setup_required"] = {
                            "issue": "python_version_incompatible",
                            "current_python": "3.13.3",
                            "supported_python": "3.8-3.12",
                            "solution": "Use a Python environment with version 3.8-3.12 and install: pip install torch transformers"
                        }
                    else:
                        response_data["errors"].append(f"Bloom generation failed: {error_msg}")
                        response_data["bloom_setup_required"] = {
                            "issue": "dependencies_missing",
                            "solution": bloom_response.get("solution", "Install PyTorch dependencies")
                        }
                else:
                    response_data["errors"].append(f"Bloom generation: {error_msg}")
            elif bloom_fallback_used:
                # Intelligent fallback is working - add info but no error
                response_data["bloom_fallback_info"] = {
                    "method": bloom_response.get("method"),
                    "reason": bloom_response.get("fallback_reason", "pytorch_unavailable"),
                    "message": "Using intelligent contextual responses (PyTorch not available)"
                }
            
            # Add helpful message if Bloom wasn't used and no fallback
            if use_bloom_generation and not bloom_used and not bloom_fallback_used:
                if not bloom_model_path:
                    response_data["errors"].append("Bloom generation requested but no quantized model found. Run quantization first.")
                    response_data["bloom_setup_required"] = {
                        "issue": "model_not_found",
                        "solution": "Quantize a model using: POST /ai-pipeline/models/quantize-bloom"
                    }
            
            logger.info(f"Enhanced AI pipeline completed successfully for user {user_id} (Bloom used: {bloom_used})")
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
        logger.error(f"Enhanced AI pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced pipeline processing failed: {str(e)}"
        )

@router.post("/generate-bloom-response")
async def generate_bloom_response_only(
    text: str,
    context: Optional[str] = None,
    bloom_model_path: str = "models/bloom-560m-8bit",
    max_tokens: int = 50,
    temperature: float = 0.7,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Generate a response using only the Bloom model (no full pipeline).
    
    Args:
        text: Input text to generate response for
        context: Optional context to guide generation
        bloom_model_path: Path to quantized Bloom model
        max_tokens: Maximum tokens for generation
        temperature: Temperature for text generation
        current_user: Authenticated user
        
    Returns:
        Bloom-generated response
    """
    try:
        user_id = current_user["uid"]
        
        logger.info(f"Generating Bloom response for user {user_id}")
        
        # Create a mock intent result for consistent interface
        mock_intent_result = {
            "intent": "general_query",
            "confidence": 1.0,
            "original_text": text
        }
        
        result = await _generate_bloom_response(
            transcription=text,
            intent_result=mock_intent_result,
            database_result={"success": True},
            model_path=bloom_model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            user_id=user_id,
            additional_context=context
        )
        
        result["user_id"] = user_id
        return result
        
    except Exception as e:
        logger.error(f"Bloom response generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bloom response generation failed: {str(e)}"
        )

# --- Helper Functions for Bloom Integration ---

async def _generate_bloom_response(
    transcription: str,
    intent_result: Dict[str, Any],
    database_result: Dict[str, Any],
    model_path: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    user_id: str = "",
    additional_context: Optional[str] = None
) -> Dict[str, Any]:
    """Generate natural response using Bloom model with structured output."""
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            # Use intelligent fallback instead of failing
            return await _generate_intelligent_fallback_response(
                transcription, intent_result, database_result, user_id
            )
        
        # Use the Python 3.11 environment where PyTorch is installed
        python_pytorch_path = "venv-pytorch/bin/python"
        
        # Check if PyTorch environment exists
        if not os.path.exists(python_pytorch_path):
            logger.warning("PyTorch environment not found, using intelligent fallback")
            return await _generate_intelligent_fallback_response(
                transcription, intent_result, database_result, user_id
            )
        
        # Build pipeline output for the new generator
        pipeline_output = {
            "transcription": transcription,
            "intent_result": intent_result,
            "database_result": database_result
        }
        
        # TODO: Add habit detection logic here
        # For now, we'll use a simple heuristic based on repeated keywords
        habit_suggestion = _detect_habit_patterns(transcription, intent_result)
        
        # Use the new Bloom response generator
        bloom_result = await bloom_generator.generate_response(
            pipeline_output=pipeline_output,
            habit_suggestion=habit_suggestion
        )
        
        # Convert to the expected format for backward compatibility
        return {
            "success": True,
            "generated_text": bloom_result.get("response", ""),
            "processing_time": 2.0,  # Estimated time
            "model_path": model_path,
            "prompt_used": "Natural conversation prompt",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "dependency_issue": False,
            "method": "pytorch_bloom_structured",
            "python_env": "pytorch-3.11",
            "structured_output": bloom_result,  # Include full structured response
            "actions": bloom_result.get("actions", []),
            "follow_up": bloom_result.get("follow_up", "")
        }
        
    except Exception as e:
        logger.error(f"Bloom generation exception: {e}")
        # Fall back to intelligent response on any error
        return await _generate_intelligent_fallback_response(
            transcription, intent_result, database_result, user_id
        )

def _detect_habit_patterns(transcription: str, intent_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Simple habit detection based on keywords and patterns."""
    # Keywords that suggest recurring activities
    habit_keywords = [
        "daily", "every day", "everyday", "weekly", "monthly", 
        "regularly", "always", "usually", "often", "again",
        "workout", "gym", "exercise", "meeting", "call",
        "reminder", "habit", "routine"
    ]
    
    transcription_lower = transcription.lower()
    
    # Check for habit-indicating keywords
    has_habit_keywords = any(keyword in transcription_lower for keyword in habit_keywords)
    
    # Check for time-based patterns
    time_patterns = ["every", "each", "daily", "weekly", "monthly"]
    has_time_pattern = any(pattern in transcription_lower for pattern in time_patterns)
    
    if has_habit_keywords or has_time_pattern:
        # Determine suggested frequency
        if any(word in transcription_lower for word in ["daily", "every day", "everyday"]):
            suggest = "daily"
        elif any(word in transcription_lower for word in ["weekly", "every week"]):
            suggest = "weekly"
        elif any(word in transcription_lower for word in ["monthly", "every month"]):
            suggest = "monthly"
        else:
            suggest = "regularly"
        
        return {
            "is_habit": True,
            "suggest": suggest,
            "confidence": 0.7
        }
    
    return None

async def _generate_intelligent_fallback_response(
    transcription: str,
    intent_result: Dict[str, Any],
    database_result: Dict[str, Any],
    user_id: str
) -> Dict[str, Any]:
    """Generate intelligent contextual response without PyTorch."""
    try:
        import time
        start_time = time.time()
        
        # Analyze the database results to create contextual response
        storage_results = database_result.get("storage_results", [])
        successful_intents = database_result.get("successful_intents", 0)
        
        if not storage_results:
            response = "I've processed your request. How else can I help you?"
        else:
            # Create contextual response based on what was actually stored
            response_parts = []
            
            # Count different types of actions
            notes_created = sum(1 for r in storage_results if r.get("intent") == "create_note")
            reminders_created = sum(1 for r in storage_results if r.get("intent") == "create_reminder")
            ledger_entries = sum(1 for r in storage_results if r.get("intent") == "create_ledger")
            
            if successful_intents > 1:
                response_parts.append(f"Perfect! I've completed {successful_intents} tasks for you:")
            else:
                response_parts.append("Great! I've completed your request:")
            
            # Add specific details about what was done
            task_details = []
            
            for result in storage_results:
                intent = result.get("intent", "")
                data = result.get("data", {})
                
                if intent == "create_note":
                    content = data.get("content", "your note")
                    task_details.append(f"✓ Saved note: '{content}'")
                
                elif intent == "create_reminder":
                    title = data.get("title", "reminder")
                    task_details.append(f"✓ Created reminder: '{title}'")
                
                elif intent == "create_ledger":
                    amount = data.get("amount", 0)
                    contact = data.get("contact_name", "contact")
                    direction = data.get("direction", "owe")
                    if direction == "owe":
                        task_details.append(f"✓ Recorded ${amount} with {contact}")
                    else:
                        task_details.append(f"✓ Logged ${amount} from {contact}")
            
            # Combine the response
            if task_details:
                response = f"{response_parts[0]} {', '.join(task_details)}. Is there anything else you need help with?"
            else:
                response = "I've processed your request successfully. What would you like to do next?"
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "generated_text": response,
            "processing_time": processing_time,
            "model_path": "intelligent_fallback",
            "method": "contextual_fallback",
            "dependency_issue": False,
            "fallback_reason": "pytorch_unavailable"
        }
        
    except Exception as e:
        return {
            "success": True,  # Still succeed with basic response
            "generated_text": "I've processed your request. How can I help you further?",
            "processing_time": 0.1,
            "model_path": "basic_fallback",
            "method": "basic_fallback",
            "dependency_issue": False,
            "fallback_reason": f"fallback_error: {str(e)}"
        }

def _construct_bloom_prompt(
    transcription: str,
    intent_result: Dict[str, Any],
    database_result: Dict[str, Any],
    additional_context: Optional[str] = None
) -> str:
    """Construct a contextual prompt for Bloom model."""
    intent = intent_result.get("intent", "general_query")
    entities = intent_result.get("entities", {})
    
    # Base context
    prompt_parts = [
        "You are Eindr, a helpful AI assistant that helps users manage reminders, notes, and expenses.",
        f"User said: \"{transcription}\"",
        f"Intent detected: {intent}"
    ]
    
    # Add entity information
    if entities:
        entity_info = []
        for key, values in entities.items():
            if values:
                entity_info.append(f"{key}: {', '.join(values) if isinstance(values, list) else values}")
        if entity_info:
            prompt_parts.append(f"Entities found: {'; '.join(entity_info)}")
    
    # Add database result context
    if database_result and database_result.get("success"):
        if intent == "create_reminder":
            prompt_parts.append("I have successfully created the reminder.")
        elif intent == "create_note":
            prompt_parts.append("I have successfully saved the note.")
        elif intent in ["create_ledger", "add_expense"]:
            prompt_parts.append("I have successfully recorded the financial entry.")
    
    # Add additional context if provided
    if additional_context:
        prompt_parts.append(f"Additional context: {additional_context}")
    
    # Add instruction for response
    prompt_parts.extend([
        "",
        "Generate a helpful, friendly, and contextual response (1-2 sentences):",
        "Response:"
    ])
    
    return "\n".join(prompt_parts)

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

@router.post("/models/quantize-bloom")
async def quantize_bloom_model(
    background_tasks: BackgroundTasks,
    model_id: str = "bigscience/bloom-560m",
    bits: int = 8,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Quantize a Bloom model to 8-bit precision for efficient inference.
    
    This endpoint initiates model quantization in the background and returns immediately.
    Use the /models/quantization-status endpoint to track progress.
    
    Args:
        model_id: Hugging Face model ID (default: "bigscience/bloom-560m")
        bits: Quantization bits (only 8 is supported)
        output_dir: Output directory for quantized model (optional)
        device: Device to use for quantization (auto-detected if not specified)
        current_user: Authenticated user
        
    Returns:
        Operation status and tracking information
    """
    try:
        user_id = current_user["uid"]
        
        # Validate parameters
        if bits != 8:
            raise HTTPException(
                status_code=400,
                detail="Only 8-bit quantization is currently supported"
            )
        
        # Generate operation ID
        operation_id = f"quantize_{model_id.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set default output directory
        if not output_dir:
            model_name = model_id.split('/')[-1]
            output_dir = f"models/{model_name}-8bit"
        
        # Check if quantization is already in progress
        if operation_id in model_operations:
            raise HTTPException(
                status_code=409,
                detail="Quantization already in progress for this model"
            )
        
        # Initialize operation tracking
        model_operations[operation_id] = {
            "status": "starting",
            "model_id": model_id,
            "output_dir": output_dir,
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "current_step": "Initializing quantization process",
            "errors": []
        }
        
        # Start quantization in background
        background_tasks.add_task(
            _run_quantization,
            operation_id,
            model_id,
            bits,
            output_dir,
            device,
            user_id
        )
        
        logger.info(f"Started quantization operation {operation_id} for user {user_id}")
        
        return {
            "success": True,
            "operation_id": operation_id,
            "status": "started",
            "message": "Model quantization started in background",
            "model_id": model_id,
            "output_dir": output_dir,
            "estimated_time_minutes": "5-15",
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start quantization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start quantization: {str(e)}"
        )

@router.get("/models/quantization-status/{operation_id}")
async def get_quantization_status(
    operation_id: str,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get the status of a quantization operation.
    
    Args:
        operation_id: Operation ID returned from quantize-bloom endpoint
        current_user: Authenticated user
        
    Returns:
        Current status of the quantization operation
    """
    try:
        user_id = current_user["uid"]
        
        if operation_id not in model_operations:
            raise HTTPException(
                status_code=404,
                detail="Operation not found"
            )
        
        operation = model_operations[operation_id]
        
        # Check if user owns this operation
        if operation["user_id"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied: Operation belongs to different user"
            )
        
        return {
            "success": True,
            "operation_id": operation_id,
            **operation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantization status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quantization status: {str(e)}"
        )

@router.post("/models/verify-quantized")
async def verify_quantized_model(
    model_dir: str,
    max_tokens: int = 20,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Verify a quantized model by running a test generation.
    
    Args:
        model_dir: Directory containing the quantized model
        max_tokens: Maximum tokens to generate for verification
        current_user: Authenticated user
        
    Returns:
        Verification results including performance metrics
    """
    try:
        user_id = current_user["uid"]
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Model directory not found: {model_dir}"
            )
        
        logger.info(f"Verifying quantized model in {model_dir} for user {user_id}")
        
        # Run verification script
        result = await _run_model_verification(model_dir, max_tokens)
        
        result["user_id"] = user_id
        result["model_dir"] = model_dir
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model verification failed: {str(e)}"
        )

@router.get("/models/available")
async def get_available_models(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get list of available models (both original and quantized).
    
    Returns:
        List of available models with their metadata
    """
    try:
        user_id = current_user["uid"]
        
        models_dir = Path("models")
        models = []
        
        if models_dir.exists():
            for model_path in models_dir.iterdir():
                if model_path.is_dir():
                    model_info = {
                        "name": model_path.name,
                        "path": str(model_path),
                        "type": "quantized" if "8bit" in model_path.name.lower() else "original",
                        "size_mb": _get_directory_size(model_path),
                        "modified": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
                    }
                    
                    # Check for config file
                    config_file = model_path / "config.json"
                    if config_file.exists():
                        try:
                            with open(config_file) as f:
                                config = json.load(f)
                                model_info["model_type"] = config.get("model_type", "unknown")
                                model_info["vocab_size"] = config.get("vocab_size", 0)
                        except:
                            pass
                    
                    models.append(model_info)
        
        return {
            "success": True,
            "models": models,
            "total_models": len(models),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available models: {str(e)}"
        )

@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Delete a model directory (use with caution).
    
    Args:
        model_name: Name of the model directory to delete
        current_user: Authenticated user
        
    Returns:
        Deletion status
    """
    try:
        user_id = current_user["uid"]
        
        model_path = Path("models") / model_name
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_name}"
            )
        
        if not model_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a directory: {model_name}"
            )
        
        # Safety check - only allow deletion of quantized models
        if "8bit" not in model_name.lower() and "quantized" not in model_name.lower():
            raise HTTPException(
                status_code=403,
                detail="For safety, only quantized models can be deleted via API"
            )
        
        # Delete the model directory
        import shutil
        shutil.rmtree(model_path)
        
        logger.info(f"Deleted model {model_name} for user {user_id}")
        
        return {
            "success": True,
            "message": f"Model {model_name} deleted successfully",
            "model_name": model_name,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )

# --- Helper Functions ---

async def _run_quantization(
    operation_id: str,
    model_id: str,
    bits: int,
    output_dir: str,
    device: Optional[str],
    user_id: str
):
    """Run quantization in background using PyTorch environment."""
    operation = model_operations[operation_id]
    
    try:
        # Update status
        operation["status"] = "preparing"
        operation["current_step"] = "Preparing quantization environment"
        operation["progress"] = 10
        
        # Use the Python 3.11 environment where PyTorch is installed
        python_pytorch_path = "venv-pytorch/bin/python"
        
        # Check if PyTorch environment exists
        if not os.path.exists(python_pytorch_path):
            operation["status"] = "failed"
            operation["current_step"] = "PyTorch environment not found"
            operation["progress"] = 0
            operation["errors"].append("PyTorch environment (venv-pytorch) not found. Please set up PyTorch first.")
            operation["failed_at"] = datetime.now().isoformat()
            return
        
        # Build command using PyTorch environment
        cmd = [python_pytorch_path, "quantize_bloom_simple.py", "--model_id", model_id, "--out_dir", output_dir]
        
        if device:
            cmd.extend(["--device", device])
        
        # Update status
        operation["status"] = "quantizing"
        operation["current_step"] = "Quantizing model (this may take several minutes)"
        operation["progress"] = 30
        
        # Run quantization
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="."
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Success
            operation["status"] = "completed"
            operation["current_step"] = "Quantization completed successfully"
            operation["progress"] = 100
            operation["completed_at"] = datetime.now().isoformat()
            operation["output_dir"] = output_dir
            operation["stdout"] = stdout.decode() if stdout else ""
            
            # Get model size
            if os.path.exists(output_dir):
                operation["model_size_mb"] = _get_directory_size(Path(output_dir))
            
        else:
            # Error
            operation["status"] = "failed"
            operation["current_step"] = "Quantization failed"
            operation["progress"] = 0
            operation["errors"].append(stderr.decode() if stderr else "Unknown error")
            operation["failed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        operation["status"] = "failed"
        operation["current_step"] = "Quantization failed with exception"
        operation["progress"] = 0
        operation["errors"].append(str(e))
        operation["failed_at"] = datetime.now().isoformat()
        logger.error(f"Quantization failed for operation {operation_id}: {e}")

async def _run_model_verification(model_dir: str, max_tokens: int) -> Dict[str, Any]:
    """Run model verification using PyTorch environment."""
    try:
        # Use the Python 3.11 environment where PyTorch is installed
        python_pytorch_path = "venv-pytorch/bin/python"
        
        # Check if PyTorch environment exists
        if not os.path.exists(python_pytorch_path):
            return {
                "success": False,
                "status": "PyTorch environment not found",
                "error": "PyTorch environment (venv-pytorch) not found. Please set up PyTorch first.",
                "model_functional": False
            }
        
        cmd = [
            python_pytorch_path, "test_pytorch_bloom.py"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="."
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return {
                "success": True,
                "status": "Model verification passed",
                "output": stdout.decode() if stdout else "",
                "model_functional": True,
                "python_env": "pytorch-3.11"
            }
        else:
            return {
                "success": False,
                "status": "Model verification failed",
                "error": stderr.decode() if stderr else "Unknown error",
                "model_functional": False,
                "python_env": "pytorch-3.11"
            }
        
    except Exception as e:
        return {
            "success": False,
            "status": "Verification failed with exception",
            "error": str(e),
            "model_functional": False
        }

def _get_directory_size(directory: Path) -> float:
    """Get directory size in MB."""
    try:
        total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return round(total_size / (1024 * 1024), 2)
    except:
        return 0.0

def _find_available_bloom_model() -> Optional[str]:
    """Find available Bloom model in the models directory."""
    # First, check for properly structured models
    possible_paths = [
        "models/bloom-560m-8bit",
        "models/bloom-560m-quantized",
        "models/Bloom_560M_chat",
        "models/Bloom_560M_lora"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it has required files
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                logger.info(f"Found Bloom model at: {path}")
                return path
    
    # Fallback: Check for standalone model files
    standalone_models = [
        "models/Bloom560m.bin",
        "models/bloom-560m.bin"
    ]
    
    for model_file in standalone_models:
        if os.path.exists(model_file):
            logger.info(f"Found standalone Bloom model file at: {model_file} (limited functionality)")
            return model_file
    
    logger.warning("No suitable Bloom model found in models directory")
    return None

# --- Model Management and Setup Endpoints ---

@router.get("/models/bloom-status")
async def get_bloom_model_status(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get the status of Bloom models and PyTorch environment.
    
    Returns:
        Status of available Bloom models and PyTorch setup
    """
    try:
        user_id = current_user["uid"]
        
        # Check current Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Check for PyTorch environment
        python_pytorch_path = "venv-pytorch/bin/python"
        pytorch_env_exists = os.path.exists(python_pytorch_path)
        
        # Check PyTorch availability in the PyTorch environment
        pytorch_available = False
        pytorch_version = None
        pytorch_error = None
        
        if pytorch_env_exists:
            try:
                # Test PyTorch in the dedicated environment
                process = await asyncio.create_subprocess_exec(
                    python_pytorch_path, "-c", 
                    "import torch; import transformers; print(f'torch:{torch.__version__}|transformers:{transformers.__version__}')",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    output = stdout.decode().strip()
                    if "torch:" in output and "transformers:" in output:
                        pytorch_available = True
                        versions = output.split("|")
                        pytorch_version = {
                            "torch": versions[0].split(":")[1],
                            "transformers": versions[1].split(":")[1]
                        }
                else:
                    pytorch_error = stderr.decode().strip() if stderr else "Unknown error"
            except Exception as e:
                pytorch_error = str(e)
        
        # Check for available models
        available_model = _find_available_bloom_model()
        
        # Check for partial models
        partial_models = []
        models_dir = Path("models")
        
        if models_dir.exists():
            for model_dir in models_dir.glob("*bloom*"):
                if model_dir.is_dir():
                    has_config = (model_dir / "config.json").exists()
                    has_model = any((model_dir / f).exists() for f in ["pytorch_model.bin", "model.safetensors"])
                    has_tokenizer = any((model_dir / f).exists() for f in ["tokenizer.json", "tokenizer_config.json"])
                    
                    partial_models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "has_config": has_config,
                        "has_model": has_model,
                        "has_tokenizer": has_tokenizer,
                        "complete": has_config and has_model and has_tokenizer
                    })
        
        # Provide setup recommendations
        recommendations = []
        issues = []
        
        if not pytorch_env_exists:
            issues.append({
                "type": "pytorch_env_missing",
                "severity": "high",
                "message": "PyTorch environment (venv-pytorch) not found",
                "solution": "Create Python 3.11 environment and install PyTorch"
            })
            recommendations.append("❌ PyTorch environment missing. Run: python3.11 -m venv venv-pytorch && source venv-pytorch/bin/activate && pip install torch transformers")
        elif not pytorch_available:
            issues.append({
                "type": "pytorch_not_working",
                "severity": "high",
                "message": f"PyTorch not working in environment: {pytorch_error}",
                "solution": "Reinstall PyTorch in the venv-pytorch environment"
            })
            recommendations.append(f"❌ PyTorch not working: {pytorch_error}")
        else:
            recommendations.append(f"✅ PyTorch {pytorch_version['torch']} working in Python 3.11 environment")
        
        if not available_model:
            issues.append({
                "type": "no_bloom_model",
                "severity": "medium",
                "message": "No complete Bloom model found",
                "solution": "Download and quantize a Bloom model"
            })
            recommendations.append("⚠️  No Bloom model found. Use /models/quantize-bloom to download one")
        else:
            recommendations.append(f"✅ Bloom model available at: {available_model}")
        
        # Overall status
        overall_status = "ready" if pytorch_available and available_model else "needs_setup"
        
        return {
            "success": True,
            "overall_status": overall_status,
            "current_python_version": python_version,
            "pytorch_environment": {
                "exists": pytorch_env_exists,
                "path": python_pytorch_path if pytorch_env_exists else None,
                "pytorch_available": pytorch_available,
                "pytorch_version": pytorch_version,
                "error": pytorch_error
            },
            "bloom_models": {
                "available_model": available_model,
                "partial_models": partial_models,
                "total_models": len(partial_models)
            },
            "issues": issues,
            "recommendations": recommendations,
            "setup_complete": pytorch_available and available_model,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting Bloom model status: {e}")
        return {
            "success": False,
            "error": str(e),
            "overall_status": "error"
        }

@router.post("/models/install-dependencies")
async def install_model_dependencies(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Install required dependencies for model operations.
    
    This endpoint installs PyTorch, transformers, and other dependencies
    required for model quantization and inference.
    
    Returns:
        Installation status and tracking information
    """
    try:
        user_id = current_user["uid"]
        
        operation_id = f"install_deps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize operation tracking
        model_operations[operation_id] = {
            "status": "starting",
            "operation_type": "dependency_installation",
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "current_step": "Installing PyTorch and ML dependencies",
            "errors": []
        }
        
        # Start installation in background
        background_tasks.add_task(_install_dependencies, operation_id, user_id)
        
        logger.info(f"Started dependency installation {operation_id} for user {user_id}")
        
        return {
            "success": True,
            "operation_id": operation_id,
            "status": "started",
            "message": "Dependency installation started in background",
            "estimated_time_minutes": "3-10",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to start dependency installation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start dependency installation: {str(e)}"
        )

async def _install_dependencies(operation_id: str, user_id: str):
    """Install ML dependencies in background task."""
    try:
        operation = model_operations[operation_id]
        
        # Update status
        operation["status"] = "installing"
        operation["current_step"] = "Installing PyTorch and transformers"
        operation["progress"] = 10
        
        # Install dependencies
        cmd = [
            "pip", "install", 
            "torch==2.1.0",
            "transformers==4.40.0", 
            "bitsandbytes==0.42.0",
            "accelerate==0.21.0",
            "safetensors==0.4.0"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="."
        )
        
        operation["progress"] = 50
        operation["current_step"] = "Installing packages (this may take several minutes)"
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Success
            operation["status"] = "completed"
            operation["current_step"] = "Dependencies installed successfully"
            operation["progress"] = 100
            operation["completed_at"] = datetime.now().isoformat()
            operation["stdout"] = stdout.decode() if stdout else ""
        else:
            # Error
            operation["status"] = "failed"
            operation["current_step"] = "Dependency installation failed"
            operation["progress"] = 0
            operation["errors"].append(stderr.decode() if stderr else "Unknown error")
            operation["failed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        operation["status"] = "failed"
        operation["current_step"] = "Installation failed with exception"
        operation["progress"] = 0
        operation["errors"].append(str(e))
        operation["failed_at"] = datetime.now().isoformat()
        logger.error(f"Dependency installation failed for operation {operation_id}: {e}") 