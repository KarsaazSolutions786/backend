"""
AI Pipeline Service for Eindr Backend
Coordinates STT, Intent Classification, Database Operations, and TTS
"""

import os
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from core.config import settings
from services.whisper_stt_service import WhisperSTTService
from services.minilm_intent_service import MiniLMIntentService
from services.database_integration_service import DatabaseIntegrationService
from services.coqui_tts_service import CoquiTTSService
from services.chat_service import ChatService
from utils.logger import logger
import time

class AIPipelineService:
    """Service that coordinates the complete AI pipeline."""
    
    def __init__(self):
        # Initialize core services
        self.whisper_stt = WhisperSTTService()
        self.minilm_intent = MiniLMIntentService()
        self.database_service = DatabaseIntegrationService()
        self.coqui_tts = CoquiTTSService()
        self.chat_service = ChatService()  # Add chat service
        
        # Pipeline statistics
        self.pipeline_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "stt_success_rate": 0.0,
            "intent_success_rate": 0.0,
            "tts_success_rate": 0.0,
            "chat_requests": 0,  # Track chat requests
            "chat_success_rate": 0.0
        }
        
        logger.info("AI Pipeline Service initialized with all components including ChatService")
    
    async def process_complete_pipeline(
        self, 
        audio_file_path: str, 
        user_id: str,
        language: str = "en",
        multi_intent: bool = True,
        store_in_database: bool = True,
        generate_audio_response: bool = True,
        voice: str = "default"
    ) -> Dict[str, Any]:
        """
        Process complete AI pipeline: STT → Intent → Database → TTS
        
        Args:
            audio_file_path: Path to audio file
            user_id: User identifier
            language: Language code for STT
            multi_intent: Enable multi-intent processing
            store_in_database: Store results in database
            generate_audio_response: Generate TTS response
            voice: Voice for TTS
            
        Returns:
            Dictionary containing complete pipeline results
        """
        start_time = time.time()
        
        try:
            self.pipeline_stats["total_requests"] += 1
            
            # Step 1: Speech-to-Text
            logger.info(f"Starting STT for user {user_id}")
            stt_result = await self.whisper_stt.transcribe_audio(audio_file_path, language)
            
            if not stt_result.get("success", False):
                self.pipeline_stats["failed_requests"] += 1
                return {
                    "success": False,
                    "error": f"STT failed: {stt_result.get('error', 'Unknown error')}",
                    "pipeline_stage": "stt"
                }
            
            transcription = stt_result.get("transcription", "")
            if not transcription:
                self.pipeline_stats["failed_requests"] += 1
                return {
                    "success": False,
                    "error": "Empty transcription",
                    "pipeline_stage": "stt"
                }
            
            logger.info(f"STT successful: '{transcription}'")
            
            # Step 2: Intent Classification
            logger.info(f"Starting intent classification for: '{transcription}'")
            intent_result = await self.minilm_intent.classify_intent(transcription, multi_intent)
            
            if not intent_result.get("success", False):
                self.pipeline_stats["failed_requests"] += 1
                return {
                    "success": False,
                    "error": f"Intent classification failed: {intent_result.get('error', 'Unknown error')}",
                    "transcription": transcription,
                    "pipeline_stage": "intent"
                }
            
            intent = intent_result.get("intent")
            confidence = intent_result.get("confidence", 0.0)
            logger.info(f"Intent classification successful: {intent} (confidence: {confidence:.2f})")
            
            # Step 3: Handle chit_chat and general_query with ChatService
            if intent in ["chit_chat", "general_query"]:
                logger.info(f"Routing {intent} to ChatService")
                self.pipeline_stats["chat_requests"] += 1
                
                # Generate conversational response
                chat_context = {
                    "intent": intent,
                    "confidence": confidence,
                    "original_transcription": transcription,
                    "processing_result": {"success": True, "data": {}}
                }
                
                try:
                    chat_response = await self.chat_service.generate_response(
                        message=transcription,
                        user_id=user_id,
                        context=chat_context
                    )
                    
                    # Generate TTS for chat response if requested
                    tts_result = None
                    if generate_audio_response:
                        try:
                            audio_data = await self.coqui_tts.synthesize_speech(chat_response, voice)
                            if audio_data:
                                tts_result = {
                                    "success": True,
                                    "response_text": chat_response,
                                    "audio_data": audio_data,
                                    "audio_size": len(audio_data),
                                    "voice_used": voice
                                }
                            else:
                                tts_result = {
                                    "success": False,
                                    "error": "TTS synthesis failed",
                                    "response_text": chat_response,
                                    "audio_size": 0,
                                    "voice_used": voice
                                }
                        except Exception as e:
                            logger.warning(f"TTS generation failed for chat response: {e}")
                            tts_result = {
                                "success": False,
                                "error": f"TTS generation failed: {str(e)}",
                                "response_text": chat_response,
                                "audio_size": 0,
                                "voice_used": voice
                            }
                    
                    processing_time = time.time() - start_time
                    self.pipeline_stats["successful_requests"] += 1
                    self._update_average_processing_time(processing_time)
                    
                    return {
                        "success": True,
                        "transcription": transcription,
                        "transcription_confidence": stt_result.get("confidence", 0.0),
                        "language_detected": language,
                        "intent_result": intent_result,
                        "tts_result": tts_result or {
                            "success": False,
                            "error": "TTS not requested",
                            "response_text": chat_response,
                            "audio_size": 0,
                            "voice_used": voice
                        },
                        "processing_time": processing_time,
                        "pipeline_stage": "chat_complete",
                        "pipeline_completed": True,
                        "chat_service_used": True,
                        "user_id": user_id
                    }
                    
                except Exception as e:
                    logger.error(f"ChatService failed: {e}")
                    # Fallback to simple response
                    fallback_response = "I understand! How can I help you stay organized today?"
                    
                    tts_result = None
                    if generate_audio_response:
                        try:
                            audio_data = await self.coqui_tts.synthesize_speech(fallback_response, voice)
                            if audio_data:
                                tts_result = {
                                    "success": True,
                                    "response_text": fallback_response,
                                    "audio_data": audio_data,
                                    "audio_size": len(audio_data),
                                    "voice_used": voice
                                }
                            else:
                                tts_result = {
                                    "success": False,
                                    "error": "TTS synthesis failed",
                                    "response_text": fallback_response,
                                    "audio_size": 0,
                                    "voice_used": voice
                                }
                        except Exception as e:
                            logger.warning(f"TTS generation failed for fallback: {e}")
                            tts_result = {
                                "success": False,
                                "error": f"TTS generation failed: {str(e)}",
                                "response_text": fallback_response,
                                "audio_size": 0,
                                "voice_used": voice
                            }
                    
                    return {
                        "success": True,
                        "transcription": transcription,
                        "transcription_confidence": stt_result.get("confidence", 0.0),
                        "language_detected": language,
                        "intent_result": intent_result,
                        "tts_result": tts_result or {
                            "success": False,
                            "error": "TTS not requested",
                            "response_text": fallback_response,
                            "audio_size": 0,
                            "voice_used": voice
                        },
                        "processing_time": time.time() - start_time,
                        "pipeline_stage": "chat_fallback",
                        "pipeline_completed": True,
                        "chat_service_used": False,
                        "user_id": user_id
                    }
            
            # Step 3: Database Operations (for other intents)
            db_result = None
            if store_in_database:
                logger.info(f"Starting database operations for intent: {intent}")
                db_result = await self._process_database_operations(intent_result, user_id)
                
                if not db_result.get("success", False):
                    logger.warning(f"Database operations failed: {db_result.get('error', 'Unknown error')}")
            
            # Step 4: Generate Response Text
            logger.info("Generating response text")
            response_text = self._generate_response_text(intent_result, db_result)
            
            # Step 5: Text-to-Speech (if requested)
            tts_result = None
            if generate_audio_response:
                logger.info("Starting TTS generation")
                tts_result = await self._generate_tts_response(intent_result, db_result, voice)
                
                if not tts_result.get("success", False):
                    logger.warning(f"TTS generation failed: {tts_result.get('error', 'Unknown error')}")
            
            # Calculate processing time and update stats
            processing_time = time.time() - start_time
            self.pipeline_stats["successful_requests"] += 1
            self._update_average_processing_time(processing_time)
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f}s")
            
            return {
                "success": True,
                "transcription": transcription,
                "transcription_confidence": stt_result.get("confidence", 0.0),
                "language_detected": language,
                "intent_result": intent_result,
                "database_result": db_result,
                "tts_result": tts_result or {
                    "success": False,
                    "error": "TTS not requested",
                    "response_text": response_text,
                    "audio_size": 0,
                    "voice_used": voice
                },
                "processing_time": processing_time,
                "pipeline_stage": "complete",
                "pipeline_completed": True,
                "chat_service_used": False,
                "user_id": user_id,
                "processing_steps": {
                    "transcription": True,
                    "intent_classification": True,
                    "database_operations": store_in_database,
                    "tts_generation": generate_audio_response
                },
                "errors": []
            }
            
        except Exception as e:
            self.pipeline_stats["failed_requests"] += 1
            logger.error(f"Pipeline processing failed: {e}")
            
            return {
                "success": False,
                "error": f"Pipeline processing failed: {str(e)}",
                "transcription": transcription if 'transcription' in locals() else "",
                "processing_time": time.time() - start_time,
                "pipeline_stage": "error",
                "user_id": user_id
            }
    
    async def _process_database_operations(self, intent_result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Process database operations based on intent classification results."""
        try:
            # Ensure user exists
            await self.database_service.create_user_if_not_exists(user_id)
            
            if intent_result.get("type") == "multi_intent":
                return await self._process_multi_intent_database(intent_result, user_id)
            else:
                return await self._process_single_intent_database(intent_result, user_id)
                
        except Exception as e:
            logger.error(f"Database operations failed: {e}")
            return {
                "success": False,
                "error": f"Database operations failed: {str(e)}"
            }
    
    async def _process_single_intent_database(self, intent_result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Process database operations for single intent."""
        try:
            intent = intent_result.get("intent")
            entities = intent_result.get("entities", {})
            confidence = intent_result.get("confidence", 0.0)
            original_text = intent_result.get("original_text", "")
            
            # Prepare classification result for database storage
            classification_result = {
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": original_text,
                "type": "single_intent"
            }
            
            # Store in database
            storage_result = await self.database_service.store_classification_result(
                classification_result=classification_result,
                user_id=user_id
            )
            
            return {
                "success": True,
                "type": "single_intent",
                "intent": intent,
                "storage_result": storage_result,
                "message": f"Successfully processed {intent} intent"
            }
            
        except Exception as e:
            logger.error(f"Single intent database processing failed: {e}")
            return {
                "success": False,
                "error": f"Single intent processing failed: {str(e)}"
            }
    
    async def _process_multi_intent_database(self, intent_result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Process database operations for multiple intents."""
        try:
            results = intent_result.get("results", [])
            overall_success = True
            processed_intents = []
            storage_results = []
            
            for result in results:
                try:
                    intent = result.get("intent")
                    entities = result.get("entities", {})
                    confidence = result.get("confidence", 0.0)
                    segment = result.get("segment", "")
                    
                    # Prepare classification result for this intent
                    classification_result = {
                        "intent": intent,
                        "confidence": confidence,
                        "entities": entities,
                        "original_text": segment,
                        "type": "multi_intent_segment"
                    }
                    
                    # Store in database
                    storage_result = await self.database_service.store_classification_result(
                        classification_result=classification_result,
                        user_id=user_id
                    )
                    
                    if storage_result.get("success", False):
                        processed_intents.append(intent)
                        storage_results.append(storage_result)
                    else:
                        overall_success = False
                        logger.error(f"Failed to store {intent}: {storage_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Failed to process intent {result}: {e}")
                    overall_success = False
            
            return {
                "success": overall_success,
                "type": "multi_intent",
                "processed_intents": processed_intents,
                "total_intents": len(results),
                "successful_intents": len(processed_intents),
                "storage_results": storage_results,
                "message": f"Processed {len(processed_intents)}/{len(results)} intents successfully"
            }
            
        except Exception as e:
            logger.error(f"Multi-intent database processing failed: {e}")
            return {
                "success": False,
                "error": f"Multi-intent processing failed: {str(e)}"
            }
    
    async def _generate_tts_response(self, intent_result: Dict[str, Any], db_result: Dict[str, Any], voice: str = "default") -> Dict[str, Any]:
        """Generate TTS response based on intent and database results."""
        try:
            response_text = self._generate_response_text(intent_result, db_result)
            
            if not response_text:
                return {
                    "success": False,
                    "error": "Failed to generate response text"
                }
            
            # Generate audio
            audio_data = await self.coqui_tts.synthesize_speech(
                text=response_text,
                voice=voice
            )
            
            if audio_data:
                return {
                    "success": True,
                    "response_text": response_text,
                    "audio_data": audio_data,
                    "audio_size": len(audio_data),
                    "voice_used": voice
                }
            else:
                return {
                    "success": False,
                    "error": "TTS synthesis failed",
                    "response_text": response_text  # Return text even if audio fails
                }
                
        except Exception as e:
            logger.error(f"TTS response generation failed: {e}")
            return {
                "success": False,
                "error": f"TTS generation failed: {str(e)}"
            }
    
    def _generate_response_text(self, intent_result: Dict[str, Any], db_result: Dict[str, Any]) -> str:
        """Generate response text based on intent and database results."""
        try:
            if intent_result.get("type") == "multi_intent":
                return self._generate_multi_intent_response(intent_result, db_result)
            else:
                return self._generate_single_intent_response(intent_result, db_result)
                
        except Exception as e:
            logger.error(f"Response text generation failed: {e}")
            return "Your request has been processed successfully!"
    
    def _generate_single_intent_response(self, intent_result: Dict[str, Any], db_result: Dict[str, Any]) -> str:
        """Generate response for single intent."""
        intent = intent_result.get("intent", "unknown")
        entities = intent_result.get("entities", {})
        
        if intent == "create_reminder":
            person = entities.get("person", [""])[0] if entities.get("person") else ""
            time = entities.get("time", [""])[0] if entities.get("time") else ""
            
            response = "Perfect! I've created a reminder"
            if person:
                response += f" to contact {person}"
            if time:
                response += f" at {time}"
            response += ". I'll make sure to notify you when the time comes!"
            return response
            
        elif intent == "create_note":
            content = entities.get("content", "")
            response = "Great! I've saved your note successfully."
            if content:
                preview = content[:30] + "..." if len(content) > 30 else content
                response += f" Your note about '{preview}' is now safely stored."
            return response
            
        elif intent in ["create_ledger", "add_expense"]:
            amount = entities.get("amount", [""])[0] if entities.get("amount") else ""
            person = entities.get("person", [""])[0] if entities.get("person") else ""
            
            response = "Excellent! I've recorded your financial entry"
            if amount and person:
                response += f" for ${amount} with {person}"
            elif amount:
                response += f" for ${amount}"
            response += ". Your records are now updated!"
            return response
            
        elif intent == "chit_chat":
            return "I understand! Is there anything specific I can help you with today? You can ask me to create reminders, notes, or track expenses."
            
        else:
            return "I've processed your request successfully! How else can I assist you today?"
    
    def _generate_multi_intent_response(self, intent_result: Dict[str, Any], db_result: Dict[str, Any]) -> str:
        """Generate response for multiple intents."""
        results = intent_result.get("results", [])
        processed_intents = db_result.get("processed_intents", [])
        
        if not results:
            return "I wasn't able to process your request. Please try again."
        
        if len(processed_intents) == 0:
            return "I encountered some issues processing your requests. Please try again or be more specific."
        
        # Generate responses for each successful intent
        responses = []
        
        for result in results:
            intent = result.get("intent")
            entities = result.get("entities", {})
            
            if intent in processed_intents:
                if intent == "create_reminder":
                    person = entities.get("person", [""])[0] if entities.get("person") else ""
                    time = entities.get("time", [""])[0] if entities.get("time") else ""
                    
                    response = "✓ Created reminder"
                    if person:
                        response += f" to contact {person}"
                    if time:
                        response += f" at {time}"
                    responses.append(response)
                    
                elif intent == "create_note":
                    content = entities.get("content", "")
                    if content:
                        preview = content[:20] + "..." if len(content) > 20 else content
                        responses.append(f"✓ Saved note: '{preview}'")
                    else:
                        responses.append("✓ Saved your note")
                        
                elif intent in ["create_ledger", "add_expense"]:
                    amount = entities.get("amount", [""])[0] if entities.get("amount") else ""
                    person = entities.get("person", [""])[0] if entities.get("person") else ""
                    
                    if amount and person:
                        responses.append(f"✓ Recorded ${amount} with {person}")
                    elif amount:
                        responses.append(f"✓ Recorded ${amount} in ledger")
                    else:
                        responses.append("✓ Added financial entry")
                        
                elif intent in ["chit_chat", "general_query"]:
                    responses.append("✓ Noted your message")
        
        # Combine responses
        if len(responses) == 1:
            return f"Perfect! {responses[0]}. Is there anything else I can help you with?"
        elif len(responses) == 2:
            return f"Excellent! I've completed both tasks: {responses[0]} and {responses[1]}. Anything else I can do for you?"
        else:
            main_response = f"Great! I've completed {len(responses)} tasks: " + ", ".join(responses[:-1]) + f", and {responses[-1]}"
            return main_response + ". Is there anything else you need help with?"
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time statistics."""
        try:
            current_avg = self.pipeline_stats["average_processing_time"]
            total_requests = self.pipeline_stats["successful_requests"]
            
            # Calculate new average
            new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
            self.pipeline_stats["average_processing_time"] = new_avg
            
        except Exception as e:
            logger.error(f"Failed to update processing time stats: {e}")
    
    async def transcribe_audio_only(self, audio_file_path: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio without intent processing."""
        try:
            result = await self.whisper_stt.transcribe_audio(audio_file_path, language)
            return result
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}",
                "transcription": "",
                "confidence": 0.0
            }
    
    async def classify_text_intent(self, text: str, multi_intent: bool = True) -> Dict[str, Any]:
        """Classify intent from text without audio processing."""
        try:
            result = await self.minilm_intent.classify_intent(text, multi_intent)
            return result
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "success": False,
                "error": f"Classification failed: {str(e)}",
                "intent": "general_query",
                "confidence": 0.0
            }
    
    async def generate_speech_audio(self, text: str, voice: str = "default") -> Optional[bytes]:
        """Generate speech audio from text."""
        try:
            audio_data = await self.coqui_tts.synthesize_speech(text, voice)
            return audio_data
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all AI services."""
        return {
            "whisper_stt": {
                "ready": self.whisper_stt.is_ready(),
                "info": self.whisper_stt.get_model_info()
            },
            "minilm_intent": {
                "ready": self.minilm_intent.is_ready(),
                "info": self.minilm_intent.get_model_info()
            },
            "tts_service": {
                "ready": self.coqui_tts.is_ready(),
                "info": self.coqui_tts.get_engine_info()
            },
            "database": {
                "ready": True,  # Database service doesn't have a ready check
                "info": "Database integration service"
            },
            "pipeline_stats": self.pipeline_stats
        }
    
    def is_ready(self) -> bool:
        """Check if the complete pipeline is ready."""
        return (
            self.whisper_stt.is_ready() and
            self.minilm_intent.is_ready() and
            self.coqui_tts.is_ready()
        ) 