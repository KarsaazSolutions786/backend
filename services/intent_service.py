from typing import Dict, List, Optional
from core.config import settings
from utils.logger import logger
import re

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
        # Multi-intent patterns
        self.multi_intent_separators = [
            r'\band\s+(?:also\s+)?(?:set|create|add|make)',  # "and set", "and also create"
            r'\balso\s+(?:set|create|add|make)',  # "also set"
            r'\bthen\s+(?:set|create|add|make)',  # "then set"
            r'\bplus\s+(?:set|create|add|make)',  # "plus set"
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
    
    async def classify_intent(self, text: str, multi_intent: bool = True) -> Dict[str, any]:
        """
        Classify the intent(s) of the given text.
        
        Args:
            text: Input text to classify
            multi_intent: If True, detect and return multiple intents; if False, return single intent (backward compatibility)
            
        Returns:
            Dictionary containing intent(s), confidence, and entities
            For multi-intent: {"intents": [...], "original_text": "..."}
            For single-intent: {"intent": "...", "confidence": 0.95, "entities": {...}, "original_text": "..."}
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("Intent model not loaded")
                if multi_intent:
                    return {"intents": [{"type": "general_query", "confidence": 0.0, "entities": {}}], "original_text": text}
                else:
                    return {"intent": "general_query", "confidence": 0.0, "entities": {}, "original_text": text}
            
            if multi_intent:
                return await self._classify_multi_intent(text)
            else:
                return await self._classify_single_intent(text)
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            if multi_intent:
                return {"intents": [{"type": "general_query", "confidence": 0.0, "entities": {}}], "original_text": text}
            else:
                return {"intent": "general_query", "confidence": 0.0, "entities": {}, "original_text": text}
    
    async def _classify_multi_intent(self, text: str) -> Dict[str, any]:
        """
        Detect and classify multiple intents in a single utterance.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with array of intents: {"intents": [...], "original_text": "..."}
        """
        logger.info(f"Multi-intent classification for: '{text}'")
        
        # Split text into segments based on multi-intent patterns
        segments = self._segment_text_for_multi_intent(text)
        
        if len(segments) <= 1:
            # Single intent detected, convert to multi-intent format
            single_result = await self._classify_single_intent(text)
            return {
                "intents": [{
                    "type": single_result["intent"],
                    "confidence": single_result["confidence"],
                    "entities": single_result["entities"],
                    "text_segment": text.strip()
                }],
                "original_text": text
            }
        
        # Process each segment
        intents = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            logger.info(f"Processing segment: '{segment}'")
            single_result = await self._classify_single_intent(segment)
            
            intents.append({
                "type": single_result["intent"],
                "confidence": single_result["confidence"],
                "entities": single_result["entities"],
                "text_segment": segment
            })
        
        logger.info(f"Multi-intent result: {len(intents)} intents detected")
        
        return {
            "intents": intents,
            "original_text": text
        }

    def _segment_text_for_multi_intent(self, text: str) -> List[str]:
        """
        Segment text into multiple intent components.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of text segments, each potentially containing a separate intent
        """
        segments = [text]  # Start with the full text
        
        # Apply multi-intent separators
        for pattern in self.multi_intent_separators:
            new_segments = []
            for segment in segments:
                # Split on the pattern but keep the separator with the second part
                parts = re.split(f'({pattern})', segment, flags=re.IGNORECASE)
                
                if len(parts) > 1:
                    # Reconstruct segments properly
                    current_segment = parts[0]
                    for i in range(1, len(parts), 2):
                        if i + 1 < len(parts):
                            # parts[i] is the separator, parts[i+1] is the content after
                            if current_segment.strip():
                                new_segments.append(current_segment.strip())
                            # Combine separator with following content
                            current_segment = parts[i] + parts[i + 1]
                        else:
                            current_segment += parts[i]
                    
                    if current_segment.strip():
                        new_segments.append(current_segment.strip())
                else:
                    new_segments.append(segment)
            
            segments = new_segments
        
        # Clean up segments - remove separators from the beginning
        cleaned_segments = []
        for segment in segments:
            # Remove leading separators
            for pattern in self.multi_intent_separators:
                segment = re.sub(f'^{pattern}\\s*', '', segment, flags=re.IGNORECASE)
            
            if segment.strip():
                cleaned_segments.append(segment.strip())
        
        logger.info(f"Text segmentation: '{text}' -> {len(cleaned_segments)} segments: {cleaned_segments}")
        return cleaned_segments

    async def _classify_single_intent(self, text: str) -> Dict[str, any]:
        """
        Classify a single intent from text (original implementation).
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing intent, confidence, and entities
        """
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
 