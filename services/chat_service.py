from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from models.models import HistoryLog
from connect_db import SessionLocal
from core.config import settings
from utils.logger import logger
import uuid
from datetime import datetime

class ChatService:
    """Chat service using Bloom 560M for conversational AI."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Bloom 560M model."""
        try:
            logger.info(f"Loading Chat model from {settings.CHAT_MODEL_PATH}")
            
            # For demo purposes, we'll simulate model loading
            # In production, uncomment and modify the following:
            """
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(settings.CHAT_MODEL_PATH)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.CHAT_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            """
            
            # Dummy model for demo
            self.model = "bloom_model_loaded"
            self.tokenizer = "bloom_tokenizer_loaded"
            
            logger.info("Chat model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Chat model: {e}")
            # For demo, continue with dummy model
            self.model = "dummy_chat_model"
            self.tokenizer = "dummy_tokenizer"
    
    async def generate_response(self, message: str, user_id: str, context: Optional[Dict] = None) -> str:
        """
        Generate a conversational response.
        
        Args:
            message: User's message
            user_id: User identifier for conversation history
            context: Additional context (intent, entities, etc.)
            
        Returns:
            Generated response text
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("Chat model not loaded")
                return "I'm sorry, I'm having trouble processing your request right now."
            
            # For demo purposes, return contextual responses
            # In production, implement actual model generation:
            """
            # Prepare conversation context
            conversation = self._get_conversation_history(user_id)
            
            # Create prompt with context
            prompt = self._create_prompt(message, conversation, context)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            """
            
            # Generate contextual dummy responses
            response = self._generate_contextual_response(message, context)
            
            # Update conversation history in database
            self._update_conversation_history(user_id, message, response)
            
            logger.info(f"Generated chat response for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Chat response generation failed: {e}")
            return "I apologize, but I encountered an error while processing your message."
    
    def _generate_contextual_response(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate contextual responses based on intent and message content."""
        message_lower = message.lower()
        
        # Check if we have intent context
        if context and "intent" in context:
            intent = context["intent"]
            
            if intent == "create_reminder":
                return "I'll help you create that reminder. Let me set that up for you right away!"
            elif intent == "create_note":
                return "Got it! I'll save that note for you. Is there anything specific you'd like me to add?"
            elif intent == "schedule_event":
                return "I can help you schedule that event. Let me check your calendar and find the best time."
            elif intent == "add_expense":
                return "I'll record that expense for you. Would you like me to categorize it or add any additional details?"
            elif intent == "list_reminders":
                return "Here are your upcoming reminders. Would you like me to modify any of them?"
            elif intent == "cancel_reminder":
                return "I'll cancel that reminder for you. Is there anything else you'd like me to help you with?"
        
        # Fallback responses based on keywords
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Eindr, your AI assistant. How can I help you manage your reminders and notes today?"
        elif any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! I'm always here to help you stay organized. Is there anything else you need?"
        elif any(word in message_lower for word in ["help", "what can you do"]):
            return "I can help you create reminders, take notes, schedule events, track expenses, and manage your tasks. Just tell me what you need!"
        elif any(word in message_lower for word in ["weather", "time", "date"]):
            return "I focus on helping you with reminders and personal organization. For weather and time information, you might want to check your device's built-in features."
        elif "?" in message:
            return "That's a great question! Based on what you're asking, I think I can help you organize that information. Would you like me to create a reminder or note about it?"
        else:
            return "I understand you're looking for assistance. Could you tell me more about what you'd like me to help you with? I'm great at managing reminders, notes, and keeping you organized!"
    
    def _get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get conversation history for a user."""
        db = SessionLocal()
        try:
            history_logs = db.query(HistoryLog).filter(
                HistoryLog.user_id == user_id,
                HistoryLog.interaction_type == "chat"
            ).order_by(HistoryLog.created_at.desc()).limit(20).all()
            
            conversations = []
            for log in reversed(history_logs):  # Reverse to get chronological order
                # Parse content which should be in format "User: message\nAssistant: response"
                if log.content and "\nAssistant:" in log.content:
                    parts = log.content.split("\nAssistant:", 1)
                    user_msg = parts[0].replace("User: ", "")
                    assistant_msg = parts[1]
                    conversations.append({
                        "user": user_msg,
                        "assistant": assistant_msg,
                        "timestamp": log.created_at.isoformat()
                    })
            
            return conversations[-10:]  # Keep only last 10 exchanges
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
                interaction_type="chat"
            )
            
            db.add(new_log)
            db.commit()
            
            logger.info(f"Saved chat history for user {user_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving conversation history: {e}")
        finally:
            db.close()
    
    def _create_prompt(self, message: str, conversation: List[Dict], context: Optional[Dict] = None) -> str:
        """Create a prompt for the model with conversation context."""
        prompt = "You are Eindr, a helpful AI assistant for managing reminders, notes, and personal organization.\n\n"
        
        # Add conversation history
        for exchange in conversation[-3:]:  # Last 3 exchanges
            prompt += f"User: {exchange['user']}\n"
            prompt += f"Assistant: {exchange['assistant']}\n\n"
        
        # Add current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "
        
        return prompt
    
    async def get_conversation_summary(self, user_id: str) -> str:
        """Get a summary of the conversation with a user."""
        try:
            history = self._get_conversation_history(user_id)
            if not history:
                return "No conversation history found."
            
            # Simple summary for demo
            total_exchanges = len(history)
            recent_topics = []
            
            for exchange in history[-3:]:
                if "remind" in exchange["user"].lower():
                    recent_topics.append("reminders")
                elif "note" in exchange["user"].lower():
                    recent_topics.append("notes")
                elif "schedule" in exchange["user"].lower():
                    recent_topics.append("scheduling")
            
            unique_topics = list(set(recent_topics))
            topics_str = ", ".join(unique_topics) if unique_topics else "general assistance"
            
            return f"Conversation summary: {total_exchanges} exchanges covering {topics_str}."
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return "Unable to generate conversation summary."
    
    def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user."""
        db = SessionLocal()
        try:
            db.query(HistoryLog).filter(
                HistoryLog.user_id == user_id,
                HistoryLog.interaction_type == "chat"
            ).delete()
            db.commit()
            logger.info(f"Cleared conversation history for user {user_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error clearing conversation history: {e}")
        finally:
            db.close()
    
    def is_ready(self) -> bool:
        """Check if the chat service is ready."""
        return self.model is not None and self.tokenizer is not None 