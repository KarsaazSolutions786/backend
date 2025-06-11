"""
Advanced Chat Service using Bloom-560M for Conversational AI
Direct model loading for local inference without vLLM dependency.
"""

import os
import re
import asyncio
import pickle
import json
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from models.models import HistoryLog
from connect_db import SessionLocal
from core.config import settings
from utils.logger import logger
import uuid
from datetime import datetime

# Try to import transformers for direct model usage
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available for direct model loading")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available - using enhanced fallback mode")

class ChatService:
    """Chat service using Bloom-560M for conversational AI with direct model loading."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_ready = False
        self.model_info = {}
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the chat service and load Bloom-560M model."""
        try:
            logger.info("Initializing Chat Service with Bloom-560M")
            
            # Check if we're in minimal mode (skip actual model loading)
            if settings.MINIMAL_MODE:
                logger.info("Running in minimal mode - Chat service will use fallback responses")
                self.model_ready = True
                self.model_info = {
                    "model_name": "bloom-560m-fallback",
                    "mode": "minimal",
                    "capabilities": ["basic_chat", "contextual_responses"]
                }
                return
            
            try:
                # Try to load the model directly
                if TRANSFORMERS_AVAILABLE:
                    self._initialize_bloom_model()
                else:
                    logger.warning("Transformers not available, using enhanced fallback mode")
                    self._setup_enhanced_fallback()
            except Exception as e:
                logger.error(f"Error during model initialization: {e}")
                self._setup_enhanced_fallback()
            
        except Exception as e:
            logger.error(f"Failed to initialize Chat service: {e}")
            self.model_ready = False
            self.model_info = {"error": str(e)}
    
    def _initialize_bloom_model(self):
        """Initialize Bloom-560M model with fallback."""
        try:
            # Check if we're in Railway deployment
            is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
            is_minimal = os.getenv("MINIMAL_MODE", "false").lower() == "true"
            
            if is_railway or is_minimal:
                logger.info("Railway/minimal mode detected - skipping Bloom-560M model loading to prevent timeouts")
                self.bloom_model = None
                self.bloom_tokenizer = None
                return False
            
            # Priority 1: Try fine-tuned chat model
            fine_tuned_path = "Models/Bloom_560M_chat"
            if os.path.exists(fine_tuned_path) and os.path.isdir(fine_tuned_path):
                logger.info(f"Loading fine-tuned Bloom chat model from {fine_tuned_path}")
                self.bloom_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
                self.bloom_model = AutoModelForCausalLM.from_pretrained(
                    fine_tuned_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                logger.info("✅ Fine-tuned Bloom chat model loaded successfully")
                return True
            
            # Priority 2: Try local binary model
            binary_model_path = "Models/bloom560m.bin" 
            if os.path.exists(binary_model_path):
                logger.info(f"Loading Bloom model from local binary: {binary_model_path}")
                self.bloom_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
                self.bloom_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
                
                # Load custom weights
                custom_weights = torch.load(binary_model_path, map_location='cpu')
                self.bloom_model.load_state_dict(custom_weights, strict=False)
                logger.info("✅ Local Bloom model loaded successfully")
                return True
            
            # Priority 3: Download from Hugging Face (only if explicitly enabled)
            download_enabled = os.getenv("ENABLE_MODEL_DOWNLOAD", "false").lower() == "true"
            if download_enabled:
                logger.info("Downloading Bloom-560M from Hugging Face (this may take several minutes)...")
                self.bloom_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
                self.bloom_model = AutoModelForCausalLM.from_pretrained(
                    "bigscience/bloom-560m",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                logger.info("✅ Bloom model downloaded and loaded successfully")
                return True
            else:
                logger.info("Model download disabled - Bloom features will be unavailable")
                self.bloom_model = None
                self.bloom_tokenizer = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Bloom model: {e}")
            self.bloom_model = None
            self.bloom_tokenizer = None
            return False
    
    def _setup_enhanced_fallback(self):
        """Setup enhanced fallback mode when model loading fails."""
        self.model_ready = True
        self.model_info = {
            "model_name": "bloom-560m-enhanced-fallback",
            "mode": "fallback_enhanced",
            "capabilities": ["contextual_responses", "pattern_matching", "conversational_patterns"]
        }
        logger.info("Enhanced fallback mode activated")
    
    async def generate_response(self, message: str, user_id: str, context: Optional[Dict] = None) -> str:
        """
        Generate a conversational response using Bloom-560M direct inference.
        
        Args:
            message: User's message (transcribed text)
            user_id: User identifier for conversation history
            context: Additional context (intent, entities, processing_result)
            
        Returns:
            Generated response text
        """
        try:
            if settings.MINIMAL_MODE or not self.model_ready:
                # Use enhanced fallback responses in minimal mode or when model unavailable
                response = self._generate_enhanced_contextual_response(message, context)
            elif self.model and self.tokenizer and TRANSFORMERS_AVAILABLE:
                # Use actual Bloom-560M model for direct inference
                response = await self._generate_bloom_response_direct(message, user_id, context)
            else:
                # Fallback to enhanced responses
                response = self._generate_enhanced_contextual_response(message, context)
            
            # Update conversation history
            self._update_conversation_history(user_id, message, response)
            
            logger.info(f"Generated chat response for user {user_id}: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Chat response generation failed: {e}")
            fallback_response = self._get_fallback_response(context)
            self._update_conversation_history(user_id, message, fallback_response)
            return fallback_response
    
    async def _generate_bloom_response_direct(self, message: str, user_id: str, context: Optional[Dict] = None) -> str:
        """Generate response using Bloom-560M with enhanced prompting and filtering."""
        try:
            # Check if this is a simple greeting that should use safe responses
            greeting_patterns = {
                'hey how are you': "I'm doing well, thanks for asking! How can I help you today?",
                'hi how are you': "I'm doing well, thank you! What can I assist you with?",
                'hello how are you': "I'm doing great! How can I help you stay organized today?",
                'hey': "Hey there! What would you like to work on?",
                'hi': "Hi! How can I help you today?",
                'hello': "Hello! What can I assist you with?",
                'good morning': "Good morning! Hope you're having a great day. How can I help?",
                'good afternoon': "Good afternoon! What can I help you with today?",
                'good evening': "Good evening! How can I assist you?",
                'what\'s up': "Not much, just here to help you stay organized! What would you like to work on?",
                'whats up': "Not much, just here to help you stay organized! What would you like to work on?"
            }
            
            # Check for exact or close matches to greeting patterns
            message_lower = message.lower().strip()
            for pattern, response in greeting_patterns.items():
                if pattern in message_lower or message_lower == pattern:
                    return response
            
            # For non-greeting messages, use the model but with heavy filtering
            conversation = self._get_conversation_history(user_id)
            prompt = self._create_bloom_prompt(message, conversation, context)
            
            # Limit prompt length to avoid issues
            if len(prompt) > 800:
                prompt = prompt[-800:]
            
            logger.debug(f"Generated prompt for Bloom-560M: {prompt[:200]}...")
            
            # Run inference in executor to avoid blocking the event loop
            import asyncio
            import concurrent.futures
            
            def _run_inference():
                try:
                    # Tokenize the prompt
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                    
                    # Generate response with optimized parameters for conversation
                    try:
                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs["input_ids"],
                                attention_mask=inputs.get("attention_mask"),
                                max_new_tokens=50,  # Shorter for better quality
                                min_new_tokens=5,   # Ensure minimum response
                                temperature=0.3,    # Much lower for more deterministic output
                                top_p=0.7,         # More focused token selection
                                top_k=20,          # Even more focused
                                repetition_penalty=1.3,  # Higher penalty to avoid repetition
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                num_return_sequences=1
                            )
                    
                        # Decode the response
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Extract only the new part (remove the prompt)
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].strip()
                        
                        return generated_text
                        
                    except Exception as e:
                        logger.error(f"Direct inference failed: {e}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Model inference error: {e}")
                    return None
                
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                generated_text = await loop.run_in_executor(executor, _run_inference)
            
            if generated_text:
                # Post-process the response
                processed_response = self._post_process_response(generated_text, context)
                return processed_response
            else:
                logger.warning("Direct inference failed, using enhanced fallback")
                return self._generate_enhanced_contextual_response(message, context)
                
        except Exception as e:
            logger.error(f"Bloom-560M direct inference failed: {e}")
            return self._generate_enhanced_contextual_response(message, context)
    
    def _create_bloom_prompt(self, message: str, conversation: List[Dict], context: Optional[Dict] = None) -> str:
        """Create a simplified, effective prompt for Bloom-560M with better conversation flow."""
        
        # Check if this is a simple greeting or casual conversation
        casual_patterns = ['hey', 'hi', 'hello', 'how are you', 'what\'s up', 'good morning', 'good afternoon', 'good evening']
        is_casual = any(pattern in message.lower() for pattern in casual_patterns)
        
        if is_casual:
            # Few-shot examples for casual conversation to guide the model
            few_shot_examples = """User: hey how are you
Eindr: I'm doing well, thanks for asking! How can I help you today?

User: hi there
Eindr: Hello! Great to hear from you. What can I assist you with?

User: good morning
Eindr: Good morning! Hope you're having a wonderful day. How can I help?

User: what's up
Eindr: Not much, just here to help you stay organized! What would you like to work on?

"""
            
            # Simple conversational prompt with examples
            return f"{few_shot_examples}User: {message}\nEindr:"
        
        else:
            # For task-oriented conversation, use structured prompt
            system_context = "Eindr is a helpful AI assistant for personal organization. Eindr helps with reminders, notes, and finances.\n\n"
            
            # Add recent conversation for context (last 2 exchanges)
            conversation_context = ""
            if conversation:
                for exchange in conversation[-2:]:
                    conversation_context += f"User: {exchange['user']}\nEindr: {exchange['assistant']}\n"
            
            # Add task context if available
            task_context = ""
            if context:
                intent = context.get("intent")
                if intent and intent != "chit_chat":
                    processing_result = context.get("processing_result", {})
                    if processing_result.get("success"):
                        task_context = f"[Completed: {intent}]\n"
            
            return f"{system_context}{conversation_context}{task_context}User: {message}\nEindr:"
    
    def _post_process_response(self, response: str, context: Optional[Dict] = None) -> str:
        """Post-process Bloom-560M response to ensure quality and appropriateness."""
        
        # Remove any repetition of the prompt or system text
        response = re.sub(r'^(You are Eindr|Eindr:|User:|System:).*?\n', '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove HTML/code content that shouldn't be in conversation
        response = re.sub(r'<[^>]+>', '', response)  # Remove HTML tags
        response = re.sub(r'http[s]?://[^\s]+', '', response)  # Remove URLs
        response = re.sub(r'<!DOCTYPE[^>]*>', '', response)  # Remove DOCTYPE
        response = re.sub(r'[{}()\[\]]+', '', response)  # Remove programming symbols
        
        # Check for incoherent patterns and reject the response
        incoherent_patterns = [
            r'pregnant',  # Personal/inappropriate content
            r'coffee.*girls',  # Unrelated rambling
            r'forum.*gameservice',  # Random website references
            r'html.*css',  # Code content
            r'console.*code',  # Programming content
            r'weird.*\.\.\.',  # Trailing confusion
            r'whoa.*ahh',  # Exclamations that don't make sense
        ]
        
        for pattern in incoherent_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # If incoherent content detected, return a safe fallback
                return self._get_safe_greeting_response(context)
        
        # Split into sentences and validate each one
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        valid_sentences = []
        
        for sentence in sentences:
            # Check if sentence is coherent (has basic structure)
            if (len(sentence) > 5 and 
                re.search(r'[a-zA-Z]{3,}', sentence) and  # Has words with 3+ letters
                not re.search(r'[#@$%^&*]+', sentence) and  # No special characters
                len(sentence.split()) >= 2):  # At least 2 words
                valid_sentences.append(sentence)
        
        if not valid_sentences:
            return self._get_safe_greeting_response(context)
        
        # Reconstruct response from valid sentences
        response = '. '.join(valid_sentences[:2])  # Max 2 sentences for brevity
        if response and not response.endswith('.'):
            response += '.'
        
        # Final length check
        if len(response) > 150:
            # Cut at natural boundary
            if '. ' in response[:150]:
                response = response[:response.find('. ', 50) + 1]
            else:
                response = response[:147] + "..."
        
        # Ensure minimum response quality
        if len(response.strip()) < 10:
            return self._get_safe_greeting_response(context)
        
        return response.strip()
    
    def _get_safe_greeting_response(self, context: Optional[Dict] = None) -> str:
        """Generate a safe, appropriate response for greetings when model output is incoherent."""
        # Simple, safe responses that work for most casual interactions
        safe_responses = [
            "I'm doing well, thank you! How can I help you today?",
            "Hello! I'm here to help you stay organized. What would you like to work on?",
            "Hi there! I'm ready to assist you with your tasks and reminders.",
            "Great to hear from you! How can I help you be more productive today?"
        ]
        
        # Return a contextually appropriate response
        import random
        return random.choice(safe_responses)
    
    def _generate_enhanced_contextual_response(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate responses using simple rule-based fallback when model unavailable."""
        # Check if we're in minimal/Railway mode
        is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
        is_minimal = os.getenv("MINIMAL_MODE", "false").lower() == "true"
        
        if is_railway or is_minimal:
            # Use simple rule-based responses for Railway deployment
            return self._get_simple_response(message)
        else:
            # For local development without model
            return f"[ERROR: Bloom-560M model not available. Install PyTorch and transformers to get AI-generated responses. Current message: '{message}']"
    
    def _get_fallback_response(self, context: Optional[Dict] = None) -> str:
        """NO HARDCODED FALLBACKS - Force model usage."""
        return "[ERROR: Model not loaded. Please install PyTorch to use Bloom-560M for responses.]"
    
    def _get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get conversation history for a user."""
        db = SessionLocal()
        try:
            history_logs = db.query(HistoryLog).filter(
                HistoryLog.user_id == user_id,
                HistoryLog.interaction_type == "chat"
            ).order_by(HistoryLog.created_at.desc()).limit(15).all()
            
            conversations = []
            for log in reversed(history_logs):  # Reverse to get chronological order
                # Parse content which should be in format "User: message\nAssistant: response"
                if log.content and "\nAssistant:" in log.content:
                    parts = log.content.split("\nAssistant:", 1)
                    user_msg = parts[0].replace("User: ", "").strip()
                    assistant_msg = parts[1].strip()
                    
                    if user_msg and assistant_msg:
                        conversations.append({
                            "user": user_msg,
                            "assistant": assistant_msg,
                            "timestamp": log.created_at.isoformat()
                        })
            
            return conversations[-8:]  # Keep only last 8 exchanges for context
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
        finally:
            db.close()
    
    def _update_conversation_history(self, user_id: str, message: str, response: str):
        """Update conversation history for a user."""
        db = SessionLocal()
        try:
            # Store conversation in content field as formatted text
            content = f"User: {message}\nAssistant: {response}"
            
            new_log = HistoryLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=content,
                interaction_type="chat",
                created_at=datetime.utcnow()
            )
            
            db.add(new_log)
            db.commit()
            
            logger.info(f"Saved chat history for user {user_id}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving conversation history: {e}")
        finally:
            db.close()
    
    async def get_conversation_summary(self, user_id: str) -> str:
        """Get a summary of the conversation with a user."""
        try:
            history = self._get_conversation_history(user_id)
            if not history:
                return "No conversation history found."
            
            # Create intelligent summary
            total_exchanges = len(history)
            recent_topics = []
            
            for exchange in history[-5:]:  # Look at last 5 exchanges
                user_text = exchange["user"].lower()
                if any(word in user_text for word in ["remind", "reminder", "schedule", "appointment"]):
                    recent_topics.append("reminders")
                if any(word in user_text for word in ["note", "write", "jot", "save", "record"]):
                    recent_topics.append("notes")
                if any(word in user_text for word in ["owe", "money", "dollar", "paid", "expense", "cost"]):
                    recent_topics.append("financial tracking")
                if any(word in user_text for word in ["help", "question", "how", "what"]):
                    recent_topics.append("general assistance")
            
            unique_topics = list(dict.fromkeys(recent_topics))  # Remove duplicates while preserving order
            topics_str = ", ".join(unique_topics) if unique_topics else "general conversation"
            
            return f"Conversation summary: {total_exchanges} exchanges covering {topics_str}. Recent focus on productivity and organization."
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return "Unable to generate conversation summary at this time."
    
    def clear_conversation_history(self, user_id: str) -> bool:
        """Clear conversation history for a user."""
        db = SessionLocal()
        try:
            deleted_count = db.query(HistoryLog).filter(
                HistoryLog.user_id == user_id,
                HistoryLog.interaction_type == "chat"
            ).delete()
            
            db.commit()
            logger.info(f"Cleared {deleted_count} conversation records for user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error clearing conversation history: {e}")
            return False
        finally:
            db.close()
    
    async def is_ready(self) -> bool:
        """Check if the chat service is ready."""
        return self.model_ready
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded chat model."""
        base_info = {
            "model_name": settings.CHAT_MODEL_NAME,
            "model_path": settings.CHAT_MODEL_PATH,
            "minimal_mode": settings.MINIMAL_MODE,
            "ready": self.model_ready,
            "inference_type": "direct" if (self.model and self.tokenizer) else "fallback",
            "capabilities": [
                "conversational_ai",
                "context_awareness", 
                "multi_turn_dialogue",
                "intent_based_responses",
                "conversation_history",
                "personalized_responses",
                "task_completion_acknowledgment"
            ]
        }
        
        if self.model_info:
            base_info.update(self.model_info)
        
        if not settings.MINIMAL_MODE and self.model_ready and self.model:
            base_info.update({
                "inference_engine": "transformers_direct",
                "max_tokens": settings.CHAT_MAX_TOKENS,
                "temperature": settings.CHAT_TEMPERATURE,
                "top_p": settings.CHAT_TOP_P,
                "top_k": settings.CHAT_TOP_K,
                "repetition_penalty": settings.CHAT_REPETITION_PENALTY,
                "model_loaded": True,
                "tokenizer_loaded": self.tokenizer is not None
            })
        
        return base_info
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the chat service."""
        health_status = {
            "service": "ChatService",
            "timestamp": datetime.utcnow().isoformat(),
            "minimal_mode": settings.MINIMAL_MODE,
            "overall_healthy": False
        }
        
        try:
            if settings.MINIMAL_MODE:
                health_status["overall_healthy"] = True
                health_status["mode"] = "fallback"
                health_status["message"] = "Running in minimal mode with enhanced fallback responses"
            else:
                # Check direct model loading
                if self.model and self.tokenizer and TRANSFORMERS_AVAILABLE:
                    health_status["model_status"] = {
                        "model_loaded": True,
                        "tokenizer_loaded": True,
                        "transformers_available": True,
                        "inference_ready": True
                    }
                    health_status["overall_healthy"] = True
                    health_status["mode"] = "direct_inference"
                    health_status["message"] = "Bloom-560M model ready for direct inference"
                elif TRANSFORMERS_AVAILABLE:
                    health_status["model_status"] = {
                        "model_loaded": False,
                        "tokenizer_loaded": False,
                        "transformers_available": True,
                        "inference_ready": False
                    }
                    health_status["mode"] = "degraded"
                    health_status["message"] = "Transformers available but model not loaded"
                else:
                    health_status["model_status"] = {
                        "model_loaded": False,
                        "tokenizer_loaded": False,
                        "transformers_available": False,
                        "inference_ready": False
                    }
                    health_status["mode"] = "fallback"
                    health_status["message"] = "Using enhanced fallback mode (transformers not available)"
            
            # Test database connectivity
            try:
                db = SessionLocal()
                db.execute("SELECT 1")
                db.close()
                health_status["database"] = {"ready": True}
            except Exception as e:
                health_status["database"] = {"ready": False, "error": str(e)}
            
            health_status["model_info"] = self.get_model_info()
            
        except Exception as e:
            health_status["error"] = str(e)
            health_status["message"] = "Health check failed"
        
        return health_status
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        # Clean up model resources if needed
        if hasattr(self, 'model') and self.model:
            try:
                del self.model
                del self.tokenizer
                if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass 
    
    def _get_simple_response(self, message: str) -> str:
        """Generate simple rule-based responses when Bloom model is unavailable."""
        message_lower = message.lower().strip()
        
        # Greeting responses
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(greeting in message_lower for greeting in greetings):
            return "Hello! How can I help you today? I can assist with reminders, notes, expenses, and general questions."
        
        # Reminder-related responses
        if any(word in message_lower for word in ["remind", "reminder", "schedule", "appointment"]):
            return "I'd be happy to help you set up a reminder! Please tell me what you'd like to be reminded about and when."
        
        # Note-related responses
        if any(word in message_lower for word in ["note", "write", "save", "remember"]):
            return "I can help you save notes! What would you like me to note down for you?"
        
        # Expense-related responses
        if any(word in message_lower for word in ["expense", "money", "cost", "spend", "buy", "paid"]):
            return "I can help you track expenses! Please tell me how much you spent and what it was for."
        
        # Question responses
        if message_lower.startswith(("what", "how", "when", "where", "why", "who")):
            return "That's a great question! While I'm running in minimal mode, I can help you with reminders, notes, and expense tracking. For more complex questions, you might want to try when the full AI model is available."
        
        # Help responses
        if any(word in message_lower for word in ["help", "support", "assist"]):
            return "I'm here to help! I can assist you with:\n• Setting reminders\n• Taking notes\n• Tracking expenses\n• Basic conversation\n\nWhat would you like to do?"
        
        # Thank you responses
        if any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help you with?"
        
        # Default response
        return "I understand you're trying to communicate with me. While I'm running in minimal mode, I can help you with reminders, notes, and expense tracking. What would you like to do?" 