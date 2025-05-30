from typing import Dict, List, Optional
from core.config import settings
from utils.logger import logger

class IntentService:
    """Intent classification service using MiniLM."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.intent_labels = [
            "create_reminder",
            "create_note", 
            "create_ledger",
            "schedule_event",
            "add_expense",
            "add_friend",
            "general_query",
            "cancel_reminder",
            "list_reminders",
            "update_reminder"
        ]
        self._load_model()
    
    def _load_model(self):
        """Load the intent classification model."""
        try:
            logger.info(f"Loading Intent model from {settings.INTENT_MODEL_PATH}")
            
            # For demo purposes, we'll simulate model loading
            # In production, uncomment and modify the following:
            """
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.INTENT_MODEL_PATH,
                num_labels=len(self.intent_labels)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(settings.INTENT_MODEL_PATH)
            """
            
            # Dummy model for demo
            self.model = "intent_model_loaded"
            self.tokenizer = "intent_tokenizer_loaded"
            
            logger.info("Intent classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Intent model: {e}")
            # For demo, continue with dummy model
            self.model = "dummy_intent_model"
            self.tokenizer = "dummy_tokenizer"
    
    async def classify_intent(self, text: str) -> Dict[str, any]:
        """
        Classify the intent of the given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing intent, confidence, and entities
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("Intent model not loaded")
                return {"intent": "general_query", "confidence": 0.0, "entities": {}}
            
            # For demo purposes, return dummy classification
            # In production, implement actual intent classification:
            """
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get top prediction
            predicted_class_id = predictions.argmax().item()
            confidence = predictions[0][predicted_class_id].item()
            intent = self.intent_labels[predicted_class_id]
            """
            
            # Simple keyword-based classification for demo
            text_lower = text.lower()
            
            if any(word in text_lower for word in ["remind", "reminder", "alert"]):
                intent = "create_reminder"
                confidence = 0.95
            elif any(word in text_lower for word in ["note", "write", "jot"]):
                intent = "create_note"
                confidence = 0.90
            elif any(word in text_lower for word in ["owe", "owes", "owed", "debt", "ledger", "borrowed", "lent", "payback"]):
                intent = "create_ledger"
                confidence = 0.92
            elif self._is_monetary_amount(text):
                # Handle standalone monetary amounts as potential ledger entries
                intent = "create_ledger"
                confidence = 0.75  # Lower confidence since context is missing
            elif any(word in text_lower for word in ["schedule", "appointment", "meeting"]):
                intent = "schedule_event"
                confidence = 0.88
            elif any(word in text_lower for word in ["expense", "cost", "spend", "money"]):
                intent = "add_expense"
                confidence = 0.85
            elif any(word in text_lower for word in ["friend", "contact", "person"]):
                intent = "add_friend"
                confidence = 0.80
            elif any(word in text_lower for word in ["cancel", "delete", "remove"]):
                intent = "cancel_reminder"
                confidence = 0.87
            elif any(word in text_lower for word in ["list", "show", "display"]):
                intent = "list_reminders"
                confidence = 0.82
            else:
                intent = "general_query"
                confidence = 0.60
            
            # Extract basic entities (time, date, etc.)
            entities = self._extract_entities(text)
            
            result = {
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": text
            }
            
            logger.info(f"Intent classification: {intent} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {"intent": "general_query", "confidence": 0.0, "entities": {}}
    
    def _extract_entities(self, text: str) -> Dict[str, any]:
        """Extract entities from text (basic implementation)."""
        entities = {}
        text_lower = text.lower()
        
        # Extract time entities
        import re
        
        # Time patterns
        time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            r'\b(\d{1,2})\s*(am|pm)\b',
            r'\bat\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                entities["time"] = matches[0]
                break
        
        # Date patterns
        date_patterns = [
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                entities["date"] = matches[0]
                break
        
        # Extract names (simple approach)
        name_patterns = [
            r'\bcall\s+([A-Z][a-z]+)\b',
            r'\bmeet\s+([A-Z][a-z]+)\b',
            r'\bwith\s+([A-Z][a-z]+)\b'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                entities["person"] = matches[0]
                break
        
        return entities
    
    async def get_intent_suggestions(self, partial_text: str) -> List[str]:
        """Get intent suggestions based on partial text."""
        try:
            # Simple suggestion based on keywords
            suggestions = []
            text_lower = partial_text.lower()
            
            if "remind" in text_lower:
                suggestions.extend([
                    "Remind me to call mom at 3 PM",
                    "Remind me to take medication",
                    "Remind me about the meeting tomorrow"
                ])
            elif "note" in text_lower:
                suggestions.extend([
                    "Note: Meeting notes from today",
                    "Note: Ideas for the project",
                    "Note: Shopping list"
                ])
            elif "schedule" in text_lower:
                suggestions.extend([
                    "Schedule a meeting with John",
                    "Schedule dentist appointment",
                    "Schedule workout session"
                ])
            
            return suggestions[:3]  # Return top 3 suggestions
            
        except Exception as e:
            logger.error(f"Failed to get intent suggestions: {e}")
            return []
    
    def is_ready(self) -> bool:
        """Check if the intent service is ready."""
        return self.model is not None and self.tokenizer is not None
    
    def _is_monetary_amount(self, text: str) -> bool:
        """Check if the text represents a standalone monetary amount."""
        import re
        
        # Remove whitespace and convert to string if needed
        text_clean = str(text).strip()
        
        # Patterns for monetary amounts
        monetary_patterns = [
            r'^\$\d+(?:,\d{3})*(?:\.\d{2})?$',  # $1000, $1,000, $1000.00
            r'^\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|bucks?|usd)$',  # 1000 dollars
            r'^€\d+(?:,\d{3})*(?:\.\d{2})?$',   # €1000
            r'^£\d+(?:,\d{3})*(?:\.\d{2})?$',   # £1000
            r'^\¥\d+(?:,\d{3})*(?:\.\d{2})?$',  # ¥1000
        ]
        
        text_lower = text_clean.lower()
        
        # Check if text matches any monetary pattern
        for pattern in monetary_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Additional check for pure dollar amounts (like "$10000")
        if re.match(r'^\$\d+$', text_clean):
            return True
            
        return False
 