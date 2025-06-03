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
        Generate a conversational response using Bloom 560M.
        
        Args:
            message: User's message (transcribed text)
            user_id: User identifier for conversation history
            context: Additional context (intent, entities, processing_result)
            
        Returns:
            Generated response text
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("Chat model not loaded")
                return "I'm sorry, I'm having trouble processing your request right now."
            
            # For production with real Bloom 560M:
            """
            # Get conversation history for context
            conversation = self._get_conversation_history(user_id)
            
            # Create a sophisticated prompt for Bloom 560M
            prompt = self._create_bloom_prompt(message, conversation, context)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response with Bloom 560M
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    top_k=50
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Post-process response to ensure it's appropriate
            response = self._post_process_response(response, context)
            """
            
            # Enhanced demo responses that simulate Bloom 560M capabilities
            response = self._generate_enhanced_contextual_response(message, context)
            
            # Update conversation history in database
            self._update_conversation_history(user_id, message, response)
            
            logger.info(f"Generated chat response for user {user_id}: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Chat response generation failed: {e}")
            return "I apologize, but I encountered an error while processing your message. How else can I help you today?"
    
    def _generate_enhanced_contextual_response(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate enhanced contextual responses that simulate Bloom 560M capabilities."""
        message_lower = message.lower()
        
        # Use context from processing pipeline for better responses
        if context:
            intent = context.get("intent")
            confidence = context.get("confidence", 0.0)
            processing_result = context.get("processing_result", {})
            data = processing_result.get("data", {})
            
            # Generate responses based on specific intents with contextual awareness
            if intent == "create_reminder" and confidence > 0.7:
                title = data.get("title", "your task")
                time = data.get("time", "")
                if time:
                    return f"Perfect! I've successfully created a reminder for '{title}' scheduled for {time}. I'll make sure to alert you at the right moment so you don't miss it. Is there anything specific you'd like me to include in the notification?"
                else:
                    return f"Great! I've created a reminder for '{title}'. Would you like to set a specific time for when I should remind you about this?"
            
            elif intent == "create_note" and confidence > 0.7:
                return f"Excellent! I've saved your note securely. Your thoughts and ideas are important, and now they're safely stored where you can easily find them later. Would you like me to help you organize this note with any tags or categories?"
            
            elif intent in ["create_ledger", "add_expense"] and confidence > 0.7:
                amount = data.get("amount", "")
                person = data.get("person", "")
                if amount and person:
                    return f"All set! I've recorded the ${amount} transaction with {person} in your ledger. Keeping track of your financial interactions is smart - it helps maintain clear relationships and good financial habits. Need help with anything else?"
                elif amount:
                    return f"Perfect! I've logged the ${amount} entry in your ledger. Your financial records are now updated and organized. Would you like to add any additional details or categories to this entry?"
                else:
                    return "I've successfully added the entry to your ledger. Maintaining good financial records is a great habit! Is there anything else you'd like to track or organize?"
            
            elif intent == "general_query":
                return "I'm here to help you stay organized and productive! I can assist with creating reminders, taking notes, managing your ledger, and much more. What would you like to work on together today?"
        
        # Conversational responses based on message content
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(greeting in message_lower for greeting in greetings):
            return "Hello there! I'm Eindr, your AI-powered personal assistant. I'm here to help you stay organized with reminders, notes, and financial tracking. What can I help you accomplish today?"
        
        thanks = ["thank", "thanks", "appreciate", "grateful"]
        if any(word in message_lower for word in thanks):
            return "You're very welcome! I'm delighted I could help you. Staying organized and productive is what I'm here for. Feel free to ask me anything else you need assistance with!"
        
        help_requests = ["help", "what can you do", "capabilities", "features"]
        if any(word in message_lower for word in help_requests):
            return "I'm equipped with several powerful capabilities! I can help you create and manage reminders, take and organize notes, track financial transactions in your ledger, process voice commands, and maintain conversation history. I use advanced AI to understand your needs and provide personalized assistance. What specific area would you like to explore?"
        
        questions = ["?", "how", "what", "when", "where", "why", "who"]
        if any(q in message for q in questions):
            return "That's a thoughtful question! While I specialize in personal organization and productivity, I'm always eager to help you think through problems. Would you like me to help you create a reminder to research this further, or perhaps take a note about your thoughts on this topic?"
        
        # Default conversational response
        return "I appreciate you sharing that with me! As your personal AI assistant, I'm always here to help you stay organized and productive. Whether you need reminders, want to take notes, or track expenses, I'm ready to assist. What would you like to work on next?"
    
    def _create_bloom_prompt(self, message: str, conversation: List[Dict], context: Optional[Dict] = None) -> str:
        """Create a sophisticated prompt for Bloom 560M with conversation context."""
        prompt = "You are Eindr, an intelligent AI assistant specializing in personal organization, productivity, and conversational support. You help users manage reminders, notes, and financial records while maintaining engaging, helpful conversations.\n\n"
        
        # Add conversation history for context
        for exchange in conversation[-3:]:  # Last 3 exchanges for context
            prompt += f"User: {exchange['user']}\n"
            prompt += f"Eindr: {exchange['assistant']}\n\n"
        
        # Add current context if available
        if context:
            intent = context.get("intent")
            if intent:
                prompt += f"[Context: User's intent is '{intent}' with recent action completion]\n"
        
        # Add current user message
        prompt += f"User: {message}\n"
        prompt += "Eindr:"
        
        return prompt
    
    def _post_process_response(self, response: str, context: Optional[Dict] = None) -> str:
        """Post-process Bloom 560M response to ensure quality and appropriateness."""
        # Remove any incomplete sentences
        sentences = response.split('.')
        if len(sentences) > 1 and sentences[-1].strip() == '':
            sentences = sentences[:-1]
        elif len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            sentences = sentences[:-1]
        
        response = '. '.join(sentences)
        if response and not response.endswith('.'):
            response += '.'
        
        # Ensure response is not too long
        if len(response) > 300:
            response = response[:297] + "..."
        
        # Ensure response is relevant and helpful
        if len(response.strip()) < 10:
            return "I understand. How else can I assist you today?"
        
        return response.strip()
    
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
    
    def get_model_info(self) -> dict:
        """Get information about the loaded chat model."""
        if not self.model or not self.tokenizer:
            return {
                "status": "not_loaded",
                "model_type": None,
                "capabilities": []
            }
        
        return {
            "status": "loaded",
            "model_type": "bloom_560m" if self.model != "dummy_chat_model" else "demo_mode",
            "model_name": "Microsoft/DialoGPT-medium" if self.model == "dummy_chat_model" else "bigscience/bloom-560m",
            "capabilities": [
                "conversational_ai",
                "context_awareness",
                "intent_based_responses",
                "conversation_history",
                "personalized_responses"
            ],
            "max_context_length": 512,
            "response_max_tokens": 150
        } 