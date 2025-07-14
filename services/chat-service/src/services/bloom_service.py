import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import threading

logger = logging.getLogger(__name__)

class BloomService:
    """Service for BLOOM-560M language model with async loading"""
    
    def __init__(self, model_path: str = "/app/models/bloom-560m"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loading = False
        self._loaded = False
        self._load_error = None
        
        # Start loading the model in a background thread
        self._start_async_loading()
    
    def _start_async_loading(self):
        """Start loading the model in a background thread"""
        def load_model_async():
            try:
                self._loading = True
                logger.info(f"Starting async loading of BLOOM-560M model from {self.model_path}")
                logger.info(f"Using device: {self.device}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                self._loaded = True
                self._loading = False
                logger.info("BLOOM-560M model loaded successfully")
                
            except Exception as e:
                self._load_error = str(e)
                self._loading = False
                logger.error(f"Failed to load BLOOM model: {e}")
        
        # Start loading in background thread
        thread = threading.Thread(target=load_model_async, daemon=True)
        thread.start()
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._loaded and self.model is not None and self.tokenizer is not None
    
    def is_loading(self) -> bool:
        """Check if the model is currently loading"""
        return self._loading
    
    def get_loading_status(self) -> Dict[str, Any]:
        """Get the current loading status"""
        return {
            "loaded": self.is_loaded(),
            "loading": self.is_loading(),
            "error": self._load_error,
            "device": self.device,
            "model_path": self.model_path
        }
    
    def _create_conversation_prompt(self, message: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Create a better conversational prompt"""
        # System instruction for better conversational behavior
        system_instruction = """You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, concise, and relevant responses to customer questions and requests. You are polite, professional, and always try to be helpful."""
        
        # Build conversation context
        prompt_parts = [system_instruction]
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-5:]:  # Keep last 5 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
        
        # Add current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean and filter the generated response"""
        # Remove any remaining prompt parts
        if "User:" in response:
            response = response.split("User:")[0]
        
        # Remove any code blocks or technical artifacts
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        response = re.sub(r'PI:BLOOM.*', '', response, flags=re.DOTALL)
        
        # Clean up whitespace and newlines
        response = re.sub(r'\n+', ' ', response)
        response = re.sub(r'\s+', ' ', response)
        response = response.strip()
        
        # Limit response length
        if len(response) > 200:
            response = response[:200] + "..."
        
        return response
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate a response using BLOOM model"""
        try:
            # Check if model is loaded
            if not self.is_loaded():
                if self.is_loading():
                    return "I'm still loading my AI model. Please try again in a moment."
                elif self._load_error:
                    logger.error(f"Model failed to load: {self._load_error}")
                    return "I'm experiencing technical difficulties. Please try again later."
                else:
                    return "I'm initializing. Please try again in a moment."
            
            # Prepare input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            if self.device == "cpu":
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Extract only the generated part (remove the input prompt)
            response = generated_text[len(prompt):].strip()
            
            logger.info(f"Generated response: {response[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def chat_response(
        self, 
        message: str, 
        conversation_history: Optional[List[Dict]] = None,
        customer_id: str = "default_user"
    ) -> str:
        """Generate a conversational response"""
        try:
            # Check if this is a simple conversational query that should use fallback
            message_lower = message.lower().strip()
            
            # Use fallback for common conversational patterns
            if any(pattern in message_lower for pattern in [
                'how are you', 'hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                'good evening', 'what can you do', 'help', 'thank', 'thanks'
            ]):
                logger.info(f"Using fallback response for message: {message}")
                return self._get_fallback_response(message)
            
            # Create a better conversational prompt
            prompt = self._create_conversation_prompt(message, conversation_history)
            
            # Generate response with appropriate parameters for chat
            response = self.generate_response(
                prompt=prompt,
                max_length=150,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Check if the response is appropriate
            if self._is_inappropriate_response(response, message):
                logger.info(f"BLOOM generated inappropriate response, using fallback for: {message}")
                return self._get_fallback_response(message)
            
            # Fallback responses for common issues
            if not response or len(response.strip()) < 5:
                response = self._get_fallback_response(message)
            
            logger.info(f"Final chat response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            return self._get_fallback_response(message)
    
    def _is_inappropriate_response(self, response: str, original_message: str) -> bool:
        """Check if the generated response is inappropriate for the original message"""
        response_lower = response.lower()
        message_lower = original_message.lower()
        
        # Check for technical artifacts or inappropriate content
        inappropriate_patterns = [
            'pi:bloom', 'technical supporter', 'which tasks', 'job?', 
            'assign', 'entity', 'framework', 'javascript', 'microsoft',
            'requirement', 'calculation', 'traditional version'
        ]
        
        # If response contains technical artifacts, it's inappropriate
        if any(pattern in response_lower for pattern in inappropriate_patterns):
            return True
        
        # If response is too short or doesn't make sense
        if len(response.strip()) < 10:
            return True
        
        # If response doesn't seem to address the original message
        if 'how are you' in message_lower and not any(greeting in response_lower for greeting in ['well', 'good', 'fine', 'thank']):
            return True
        
        return False
    
    def _get_fallback_response(self, message: str) -> str:
        """Provide fallback responses for common questions"""
        message_lower = message.lower().strip()
        
        # Common greeting patterns
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'how are you']):
            return "Hello! I'm doing well, thank you for asking. How can I help you today?"
        
        # Common questions
        if 'how are you' in message_lower:
            return "I'm doing well, thank you! I'm here to help you with any questions or tasks you might have."
        
        if 'what can you do' in message_lower or 'help' in message_lower:
            return "I can help you with various tasks like answering questions, providing information, and engaging in conversation. What would you like to know?"
        
        if 'thank' in message_lower:
            return "You're welcome! I'm happy to help. Is there anything else you'd like to know?"
        
        # Default response
        return f"I understand you said: '{message}'. How can I help you with that?"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "BLOOM-560M",
            "model_path": self.model_path,
            "device": self.device,
            "loaded": self.is_loaded(),
            "loading": self.is_loading(),
            "error": self._load_error,
            "parameters": "560M" if self.model else "Unknown"
        }

# Global instance
bloom_service = None

def get_bloom_service() -> BloomService:
    """Get the global BLOOM service instance"""
    global bloom_service
    if bloom_service is None:
        model_path = os.getenv("BLOOM_MODEL_PATH", "/app/models/bloom-560m")
        bloom_service = BloomService(model_path)
    return bloom_service 