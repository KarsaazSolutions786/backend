from typing import Dict, List, Optional
from core.config import settings
from utils.logger import logger
import re

class IntentService:
    """Intent classification service using MiniLM."""
    
    def __init__(self, language_code: str = "en"):
        self.language_code = language_code
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
        # Language-specific resources will be loaded here
        self.multi_intent_separators: List[str] = []
        self.intent_keywords: Dict[str, List[str]] = {}
        self.entity_patterns: Dict[str, List[str]] = {}
        self.fallback_intent_indicators: List[str] = []

        self._load_language_resources(language_code)
        self._load_model()
    
    def _load_language_resources(self, lang_code: str):
        """Load language-specific keywords and patterns."""
        logger.info(f"Loading language resources for: {lang_code}")
        # Default to English if the specified language is not (yet) supported
        if lang_code == "en":
            self.multi_intent_separators = [
                r'\band\s+(?:also\s+)?(?:set|create|add|make|remind)',
                r'\balso\s+(?:set|create|add|make|remind)',
                r'\bthen\s+(?:set|create|add|make|remind)',
                r'\bplus\s+(?:set|create|add|make|remind)',
                r'\band\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owes?|owed?|borrowed?|lent)',
                r'\band\s+(?:I|i)\s+(?:owe|borrowed|lent)',
                r'\band\s+([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay)',
                r'\band\s+(?:I|i)\s+(?:want|need|have to|should)',
                r'\band\s+(?:remind|note|track|record)',
                r'\band\s+(?:\$\d+|\d+\s*dollars?)',
            ]
            
            self.intent_keywords = {
                "create_reminder": ["remind", "reminder", "alert", "book a ticket", "book ticket"],
                "create_note": ["note", "write", "jot"],
                "create_ledger": ["owe", "owes", "owed", "debt", "ledger", "borrowed", "lent", "payback", "will give", "will pay", "giving", "paying", "pay me", "give me"],
                "schedule_event": ["schedule", "appointment", "meeting"],
                "add_expense": ["expense", "cost", "spend", "money"], # Often overlaps with ledger
                "add_friend": ["friend", "contact", "person"],
                "cancel_reminder": ["cancel", "delete", "remove"],
                "list_reminders": ["list", "show", "display"],
                "general_query_phrases": ["i want to go", "want to go", "going to", "travel to", "visit"]
            }

            self.entity_patterns = {
                "time": [
                    r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
                    r'\b(\d{1,2})\s*(am|pm)\b',
                    r'\bat\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?\b'
                ],
                "date": [
                    r'\b(today|tomorrow|yesterday)\b',
                    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                    r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'
                ],
                "name_general": [
                    r'\bcall\s+([A-Z][a-z]+)\b',
                    r'\bmeet\s+([A-Z][a-z]+)\b',
                    r'\bwith\s+([A-Z][a-z]+)\b',
                ],
                "name_ledger": [
                    r'\b([A-Z][a-z]+)\s+(?:owes?|owed?|borrowed?|lent)',
                    r'(?:owes?|owed?|borrowed?|lent)\s+([A-Z][a-z]+)',
                    r'\b([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay)',
                    r'(?:give|pay)\s+([A-Z][a-z]+)',
                    r'\b([A-Z][a-z]+)\s+.*?\$\d+',
                    r'\$\d+.*?\b([A-Z][a-z]+)',
                ],
                "name_additional": [
                    r'\b([A-Z][a-z]+)\s+(?:and|&)',
                    r'(?:from|to)\s+([A-Z][a-z]+)',
                ],
                "money": [
                    r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
                    r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',
                    r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*bucks?',
                ],
                "money_standalone": [ # For _is_monetary_amount
                    r'^$\d+(?:,\d{3})*(?:\.\d{2})?$',
                    r'^(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|bucks?|usd)$',
                    # Add other currencies here, e.g., Euro, Pound, Yen
                    r'^€\d+(?:,\d{3})*(?:\.\d{2})?$',
                    r'^£\d+(?:,\d{3})*(?:\.\d{2})?$',
                    r'^¥\d+(?:,\d{3})*(?:\.\d{2})?$',
                ],
                "name_for_has_name_and_money": [ # For _has_name_and_money
                    r'\b([A-Z][a-z]+)\s+(?:owes?|owed?|borrowed?|lent)',
                    r'(?:owes?|owed?|borrowed?|lent)\s+([A-Z][a-z]+)',
                    r'\b([A-Z][a-z]+)\s+(?:\$|\d+)', # Name $amount or Name amount
                    r'(?:$|\\d+).*?([A-Z][a-z]+)', # $amount ... Name
                    r'\b([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay|owes?|owed?)',
                    r'\b([A-Z][a-z]+).*?(?:$|\\d+)',
                    r'(?:$|\\d+).*?\b([A-Z][a-z]+)',
                ]
            }
            self.fallback_intent_indicators = [
                'remind', 'note', 'owe', 'want', 'need', 'have to', 'should', 
                'i ', 'john', 'sarah', 'mike', '$', 'dollar', 'track', 'record' # Example names/terms
            ]
        else:
            # Placeholder for other languages - ideally load from JSON files or a dedicated i18n module
            logger.warning(f"Language resources for '{lang_code}' not implemented. Falling back to English defaults.")
            if lang_code != "en": # Avoid infinite recursion if "en" fails
                self._load_language_resources("en") # Fallback to English
            else: # If English itself failed, use minimal defaults to avoid crashing
                self.multi_intent_separators = [r'\band\s+'] 
                self.intent_keywords = {"general_query": ["."]} # Match anything
                self.entity_patterns = {}
                self.fallback_intent_indicators = []

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
        Classify the intent(s) of the given text using the service's configured language.
        
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
                logger.error(f"Intent model not loaded for language {self.language_code}")
                if multi_intent:
                    return {"intents": [{"type": "general_query", "confidence": 0.0, "entities": {}}], "original_text": text}
                else:
                    return {"intent": "general_query", "confidence": 0.0, "entities": {}, "original_text": text}
            
            if multi_intent:
                return await self._classify_multi_intent(text)
            else:
                return await self._classify_single_intent(text)
                
        except Exception as e:
            logger.error(f"Intent classification failed for language {self.language_code}: {e}")
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
        logger.info(f"Multi-intent classification for ('{self.language_code}'): '{text}'")
        
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
                
            logger.info(f"Processing segment ('{self.language_code}'): '{segment}'")
            single_result = await self._classify_single_intent(segment)
            
            intents.append({
                "type": single_result["intent"],
                "confidence": single_result["confidence"],
                "entities": single_result["entities"],
                "text_segment": segment
            })
        
        logger.info(f"Multi-intent result ('{self.language_code}'): {len(intents)} intents detected")
        
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
                if any(indicator in following_text for indicator in self.fallback_intent_indicators):
                    and_positions.append({
                        'start': match.start(),
                        'end': match.end(),
                        'match': match.group(),
                        'pattern': 'fallback_and'
                    })
            
            separator_positions = and_positions
        
        # If still no separators, return the original text
        if not separator_positions:
            logger.info(f"Text segmentation ('{self.language_code}'): '{text}' -> 1 segment (no separators found)")
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
        
        logger.info(f"Text segmentation ('{self.language_code}'): '{text}' -> {len(cleaned_segments)} segments: {cleaned_segments}")
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
        
        logger.info(f"Intent classification ('{self.language_code}'): {intent} (confidence: {confidence:.2f}) for text: '{text}'")
        return result

    def _extract_entities(self, text: str) -> Dict[str, any]:
        """Extract entities from text (enhanced implementation)."""
        entities = {}
        text_lower = text.lower()
        
        # Use language-specific entity patterns
        lang_entity_patterns = self.entity_patterns

        if not lang_entity_patterns: # No patterns for this language
            return entities

        # Time
        for pattern_str in lang_entity_patterns.get("time", []):
            if matches := re.findall(pattern_str, text_lower):
                entities["time"] = matches[0]; break
        # Date
        for pattern_str in lang_entity_patterns.get("date", []):
            if matches := re.findall(pattern_str, text_lower): # Note: some date words might be caught by mistake
                entities["date"] = matches[0]; break
        
        # Names (combining general, ledger, additional for simplicity here)
        # This needs more sophisticated NER for multilingual
        name_patterns_combined = \
            lang_entity_patterns.get("name_general", []) + \
            lang_entity_patterns.get("name_ledger", []) + \
            lang_entity_patterns.get("name_additional", [])
        
        for pattern_str in name_patterns_combined:
            # Names are case-sensitive, so search in original text
            if matches := re.findall(pattern_str, text): 
                entities["person"] = matches[0]; break
        
        # Money
        for pattern_str in lang_entity_patterns.get("money", []):
            if matches := re.findall(pattern_str, text_lower):
                entities["amount"] = matches[0]; break
                
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
        if self.language_code != "en": # This logic is too English-specific
            # For other languages, this would need proper NER or more generic patterns
            return False 

        lang_entity_patterns = self.entity_patterns
        has_name = any(re.search(p, text) for p in lang_entity_patterns.get("name_for_has_name_and_money",[]))
        has_money = any(re.search(p, text, re.IGNORECASE) for p in lang_entity_patterns.get("money",[]))
        return has_name and has_money

    def _is_monetary_amount(self, text: str) -> bool:
        """Check if the text represents a standalone monetary amount."""
        text_clean = str(text).strip()
        text_lower = text_clean.lower()
        
        # Uses patterns including common currency symbols, can be expanded
        money_patterns = self.entity_patterns.get("money_standalone", [])
        for pattern_str in money_patterns:
            if re.match(pattern_str, text_lower):
                return True
        
        if self.language_code == "en" and re.match(r'^$\d+$', text_clean): # English-specific quick check
            return True
            
        return False
 