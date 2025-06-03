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
        # Multi-intent patterns - Enhanced for better detection
        self.multi_intent_separators = [
            # Primary action-based separators
            r'\band\s+(?:also\s+)?(?:set|create|add|make|remind)',  # "and set", "and also create", "and remind"
            r'\balso\s+(?:set|create|add|make|remind)',  # "also set", "also remind"
            r'\bthen\s+(?:set|create|add|make|remind)',  # "then set", "then remind"
            r'\bplus\s+(?:set|create|add|make|remind)',  # "plus set", "plus remind"
            
            # Name-based separators (for ledger entries)
            r'\band\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owes?|owed?|borrowed?|lent)',  # "and John owes"
            r'\band\s+(?:I|i)\s+(?:owe|borrowed|lent)',  # "and I owe", "and i borrowed"
            r'\band\s+([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay)',  # "and John will give", "and John pay"
            
            # General "and" separators for different intent types
            r'\band\s+(?:I|i)\s+(?:want|need|have to|should)',  # "and i want", "and i need"
            r'\band\s+(?:remind|note|track|record)',  # "and remind", "and note", "and track"
            
            # Monetary separators
            r'\band\s+(?:\$\d+|\d+\s*dollars?)',  # "and $50", "and 50 dollars"
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
        Segment text into multiple intent components using enhanced logic.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of text segments, each potentially containing a separate intent
        """
        # First, try to find all separator positions
        separator_positions = []
        
        for pattern in self.multi_intent_separators:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                separator_positions.append({
                    'start': match.start(),
                    'end': match.end(),
                    'match': match.group(),
                    'pattern': pattern
                })
        
        # Sort separators by position
        separator_positions.sort(key=lambda x: x['start'])
        
        # If no separators found, try fallback patterns
        if not separator_positions:
            # Look for standalone "and" that might separate different intents
            and_positions = []
            for match in re.finditer(r'\band\s+', text, re.IGNORECASE):
                # Check if this "and" is followed by potential intent indicators
                following_text = text[match.end():match.end()+50].lower()
                if any(indicator in following_text for indicator in [
                    'remind', 'note', 'owe', 'want', 'need', 'have to', 'should', 
                    'i ', 'john', 'sarah', 'mike', '$', 'dollar', 'track', 'record'
                ]):
                    and_positions.append({
                        'start': match.start(),
                        'end': match.end(),
                        'match': match.group(),
                        'pattern': 'fallback_and'
                    })
            
            separator_positions = and_positions
        
        # If still no separators, return the original text
        if not separator_positions:
            logger.info(f"Text segmentation: '{text}' -> 1 segment (no separators found)")
            return [text]
        
        # Split text based on separator positions
        segments = []
        last_end = 0
        
        for sep in separator_positions:
            # Add text before this separator
            if sep['start'] > last_end:
                segment = text[last_end:sep['start']].strip()
                if segment:
                    segments.append(segment)
            
            # Start next segment from this separator (including it)
            last_end = sep['start']
        
        # Add remaining text after last separator
        if last_end < len(text):
            segment = text[last_end:].strip()
            if segment:
                segments.append(segment)
        
        # Clean up segments - remove leading separators and connectors
        cleaned_segments = []
        for segment in segments:
            # Remove leading "and", "also", "then", "plus" etc.
            cleaned = re.sub(r'^\s*(?:and|also|then|plus)\s+', '', segment, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            
            # Skip very short segments that are likely artifacts
            if len(cleaned) > 3:
                cleaned_segments.append(cleaned)
        
        # If we ended up with just one segment, try a more aggressive approach
        if len(cleaned_segments) <= 1:
            # Split on any "and" that appears to separate different types of content
            aggressive_segments = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
            if len(aggressive_segments) > 1:
                cleaned_segments = [seg.strip() for seg in aggressive_segments if seg.strip()]
        
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
        
        # Enhanced keyword-based classification for demo
        text_lower = text.lower()
        
        # Check for reminder keywords first
        if any(word in text_lower for word in ["remind", "reminder", "alert", "book a ticket", "book ticket"]):
            intent = "create_reminder"
            confidence = 0.95
        # Check for note keywords
        elif any(word in text_lower for word in ["note", "write", "jot"]):
            intent = "create_note"
            confidence = 0.90
        # Enhanced ledger detection - look for names + money terms or explicit debt language
        elif (any(word in text_lower for word in ["owe", "owes", "owed", "debt", "ledger", "borrowed", "lent", "payback", "will give", "will pay", "giving", "paying", "pay me", "give me"]) or
              self._has_name_and_money(text) or 
              self._is_monetary_amount(text)):
            intent = "create_ledger"
            confidence = 0.92
        # Check for scheduling keywords
        elif any(word in text_lower for word in ["schedule", "appointment", "meeting"]):
            intent = "schedule_event"
            confidence = 0.88
        # Check for expense keywords
        elif any(word in text_lower for word in ["expense", "cost", "spend", "money"]):
            intent = "add_expense"
            confidence = 0.85
        # Check for friend/contact keywords
        elif any(word in text_lower for word in ["friend", "contact", "person"]):
            intent = "add_friend"
            confidence = 0.80
        # Check for cancellation keywords
        elif any(word in text_lower for word in ["cancel", "delete", "remove"]):
            intent = "cancel_reminder"
            confidence = 0.87
        # Check for listing keywords
        elif any(word in text_lower for word in ["list", "show", "display"]):
            intent = "list_reminders"
            confidence = 0.82
        # Check for travel/general chat patterns
        elif any(phrase in text_lower for phrase in ["i want to go", "want to go", "going to", "travel to", "visit"]):
            intent = "general_query"
            confidence = 0.75
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
        """Extract entities from text (enhanced implementation)."""
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
        
        # Extract names (enhanced approach for different contexts)
        name_patterns = [
            # General conversation patterns
            r'\bcall\s+([A-Z][a-z]+)\b',
            r'\bmeet\s+([A-Z][a-z]+)\b',
            r'\bwith\s+([A-Z][a-z]+)\b',
            
            # Ledger/financial patterns
            r'\b([A-Z][a-z]+)\s+(?:owes?|owed?|borrowed?|lent)',  # "John owes", "John borrowed"
            r'(?:owes?|owed?|borrowed?|lent)\s+([A-Z][a-z]+)',     # "owes John", "borrowed from John"
            r'\b([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay)',        # "John will give", "John pay"
            r'(?:give|pay)\s+([A-Z][a-z]+)',                      # "give John", "pay John"
            r'\b([A-Z][a-z]+)\s+.*?\$\d+',                        # "John ... $50"
            r'\$\d+.*?\b([A-Z][a-z]+)',                           # "$50 ... John"
            
            # Additional name patterns
            r'\b([A-Z][a-z]+)\s+(?:and|&)',                       # "John and", "John &"
            r'(?:from|to)\s+([A-Z][a-z]+)',                       # "from John", "to John"
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                entities["person"] = matches[0]
                break
        
        # Extract monetary amounts for ledger entries
        money_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',     # $50, $1,000, $50.00
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',  # 50 dollars, 1000 dollars
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*bucks?',    # 50 bucks
        ]
        
        for pattern in money_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                entities["amount"] = matches[0]
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
    
    def _has_name_and_money(self, text: str) -> bool:
        """Check if text contains both a person's name and a monetary amount."""
        import re
        
        # Look for names (capitalized words that could be names)
        name_patterns = [
            r'\b([A-Z][a-z]+)\s+(?:owes?|owed?|borrowed?|lent)',  # "John owes"
            r'(?:owes?|owed?|borrowed?|lent)\s+([A-Z][a-z]+)',     # "owes John"
            r'\b([A-Z][a-z]+)\s+(?:\$|\d+)',                      # "John $50"
            r'(?:\$|\d+).*?([A-Z][a-z]+)',                        # "$50 John"
            # Enhanced patterns for more flexible matching
            r'\b([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay|owes?|owed?)',  # "John will give", "John will pay"
            r'\b([A-Z][a-z]+).*?(?:\$|\d+)',                      # "John ... $50" (flexible with text in between)
            r'(?:\$|\d+).*?\b([A-Z][a-z]+)',                      # "$50 ... John" (flexible with text in between)
        ]
        
        has_name = any(re.search(pattern, text) for pattern in name_patterns)
        
        # Look for monetary amounts
        money_patterns = [
            r'\$\d+',           # $50
            r'\d+\s*dollars?',  # 50 dollars
            r'\d+\s*bucks?',    # 50 bucks
        ]
        
        has_money = any(re.search(pattern, text, re.IGNORECASE) for pattern in money_patterns)
        
        return has_name and has_money

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
            r'^¥\d+(?:,\d{3})*(?:\.\d{2})?$',  # ¥1000
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
 