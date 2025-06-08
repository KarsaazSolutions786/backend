from typing import Dict, List, Optional
from core.config import settings
from utils.logger import logger
import re
import pickle
import os
import numpy as np
from pathlib import Path

# Optional PyTorch imports
try:
    # Only import transformers if not in Railway/minimal mode
    is_minimal_mode = os.getenv("MINIMAL_MODE", "false").lower() == "true"
    is_railway_env = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    if not (is_minimal_mode or is_railway_env):
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        PYTORCH_AVAILABLE = True
        logger.info("PyTorch and transformers are available")
    else:
        PYTORCH_AVAILABLE = False
        logger.info("Railway/minimal mode detected - skipping transformers import")
except ImportError as e:
    PYTORCH_AVAILABLE = False
    logger.warning(f"PyTorch not available: {e}. Will use fallback model.")

class IntentService:
    """Intent classification service using prompt engineering approach."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Updated intent labels to match the prompt categories
        self.intent_labels = [
            "create_reminder",  # Maps to "Reminders"
            "create_note",      # Maps to "Notes" 
            "create_ledger",    # Maps to "Ledger"
            "general_query"     # Maps to "Chitchat"
        ]
        
        # Category mapping for prompt engineering
        self.category_mapping = {
            "Reminders": "create_reminder",
            "Ledger": "create_ledger", 
            "Notes": "create_note",
            "Chitchat": "general_query"
        }
        
        # Enhanced multi-intent patterns for better detection
        self.multi_intent_separators = [
            # Primary action-based separators with context awareness
            r'\band\s+(?:also\s+)?(?:please\s+)?(?:set|create|add|make|remind|schedule)',
            r'\balso\s+(?:please\s+)?(?:set|create|add|make|remind|schedule)',
            r'\bthen\s+(?:please\s+)?(?:set|create|add|make|remind|schedule)',
            r'\bplus\s+(?:please\s+)?(?:set|create|add|make|remind|schedule)',
            
            # Task-specific separators
            r'\band\s+(?:remind|note|track|record|schedule)',
            r'\band\s+(?:I|i)\s+(?:want|need|have to|should|would like to)',
            
            # Name-based separators for ledger/contact entries
            r'\band\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owes?|owed?|borrowed?|lent|paid?)',
            r'\band\s+(?:I|i)\s+(?:owe|borrowed|lent|paid?)',
            
            # Monetary and time-based separators
            r'\band\s+(?:\$\d+|\d+\s*dollars?)',
            r'\band\s+(?:at|on|by|before)\s+(?:\d{1,2}(?::\d{2})?(?:\s*[ap]m)?|\d{1,2}(?:st|nd|rd|th)?)',
        ]
        self._initialize_prompt_classifier()
    
    def _initialize_prompt_classifier(self):
        """Initialize the prompt-based classifier with rules and examples."""
        logger.info("Initializing prompt-based intent classifier")
        
        # Define classification rules based on prompt engineering
        self.classification_rules = {
            "Reminders": {
                "keywords": [
                    "remind", "reminder", "alert", "alarm", "schedule", "appointment",
                    "meeting", "call", "email", "take", "medicine", "medication",
                    "doctor", "dentist", "workout", "exercise", "pick up", "drop off"
                ],
                "patterns": [
                    r'\b(remind|reminder)\b.*\b(me|to)\b',
                    r'\bset\s+(a\s+)?(reminder|alarm)\b',
                    r'\b(appointment|meeting)\b.*\b(tomorrow|today|at|on)\b',
                    r'\b(call|email|text)\b.*\b(at|tomorrow|today)\b',
                    r'\btake\b.*\b(medicine|medication|pills?)\b'
                ],
                "time_indicators": ["at", "pm", "am", "tomorrow", "today", "tonight", "morning", "evening", "o'clock"]
            },
            
            "Ledger": {
                "keywords": [
                    "owe", "owes", "owed", "borrowed", "lent", "paid", "pay", "money",
                    "dollar", "dollars", "bucks", "debt", "loan", "expense", "cost",
                    "spent", "gave", "received", "back", "return", "give"
                ],
                "patterns": [
                    r'\b(owe|owes|owed)\b',
                    r'\b(borrowed|lent)\b',
                    r'\b(paid|pay)\b.*\b(back|me|him|her)\b',
                    r'\$\d+',
                    r'\b\d+\s*(dollars?|bucks?)\b',
                    r'\b(expense|cost|spent)\b',
                    r'\b[A-Z][a-z]+\s+(?:will\s+)?(?:give|pay)\s+(?:me|us)',  # "John will give me"
                    r'\b(?:give|pay)\s+(?:me|us)\s+\$\d+',                    # "give me $50"
                    r'\b[A-Z][a-z]+.*\$\d+',                                  # "John ... $50"
                    r'\$\d+.*\b[A-Z][a-z]+',                                  # "$50 ... John"
                    r'\b(?:will\s+)?(?:give|pay).*\$\d+',                     # "will give ... $50"
                ],
                "money_indicators": ["$", "dollar", "dollars", "bucks", "money", "paid", "cost", "give", "pay"]
            },
            
            "Notes": {
                "keywords": [
                    "note", "write", "jot", "save", "record", "remember", "list",
                    "shopping", "grocery", "groceries", "buy", "purchase", "get", "idea", "thought", "memo", "diary",
                    "journal", "log", "minutes", "summary", "discussion", "add", "create", "make"
                ],
                "patterns": [
                    r'\b(add|create|make)\s+(a\s+)?(note|list)\b',  # "add a note", "create note", "make list"
                    r'\bnote\s+(to\s+|about\s+|that\s+)',  # "note to", "note about", "note that"
                    r'\b(note|write|jot)\b.*\b(down|this|that)\b',
                    r'\b(save|record|remember)\b.*\b(this|that|it)\b',
                    r'\b(shopping|grocery)\s+(list|items?)\b',
                    r'\b(meeting\s+)?minutes\b',
                    r'\b(idea|thought|discussion)\b',
                    r'\b(buy|purchase|get)\s+(groceries|food|items?)\b',  # Shopping related
                    r'\bset\s+(?:a\s+)?note\s+to\s+buy\b',  # "set a note to buy"
                ],
                "content_indicators": ["list", "idea", "thought", "discussion", "meeting", "project", "groceries", "shopping", "buy", "purchase"]
            },
            
            "Chitchat": {
                "keywords": [
                    "hello", "hi", "hey", "how", "what", "who", "when", "where", "why",
                    "can", "you", "help", "thank", "thanks", "please", "sorry",
                    "good", "morning", "evening", "night", "weather", "doing"
                ],
                "patterns": [
                    r'\b(hello|hi|hey)\b',
                    r'\bhow\s+(are\s+you|is\s+it)\b',
                    r'\bwhat\s+(can\s+you|do\s+you|is)\b',
                    r'\b(thank|thanks)\b',
                    r'\bgood\s+(morning|evening|night)\b',
                    r'\b(help|assist)\b.*\b(me|us)\b'
                ],
                "question_indicators": ["how", "what", "who", "when", "where", "why", "can", "?"]
            }
        }
        
        self.model = "prompt_based_classifier"
        self.tokenizer = "rule_based_tokenizer"
        logger.info("Prompt-based classifier initialized successfully")

    async def classify_intent(self, text: str, multi_intent: bool = True) -> Dict[str, any]:
        """
        Classify intent using prompt engineering approach.
        
        Args:
            text: Input text to classify
            multi_intent: If True, detect and return multiple intents
            
        Returns:
            Dictionary containing intent(s), confidence, and entities
        """
        try:
            # Clean and normalize input text
            text = self._normalize_text(text)
            
            if multi_intent:
                return await self._classify_multi_intent(text)
            else:
                return await self._classify_single_intent(text)
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._create_fallback_response(text, multi_intent)

    async def _classify_single_intent(self, text: str) -> Dict[str, any]:
        """Classify a single intent using prompt engineering."""
        try:
            # Create the prompt
            # prompt = f"Classify the following text into one of these categories: Reminders, Ledger, Notes, or Chitchat.\n\nText: {text}\nCategory:"
            # logger.info(f"Using prompt: {prompt}")
            
            # Analyze text using rules
            category_scores = self._analyze_text_with_rules(text)
            
            # Get the best category
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category_name = best_category[0]
            confidence = best_category[1]
            
            # Map category to intent
            intent = self.category_mapping.get(category_name, "general_query")
            
            # Extract entities
            entities = self._extract_entities(text)
            
            logger.info(f"Classification result: {category_name} -> {intent} (confidence: {confidence:.3f})")
            
            return {
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": text,
                "category": category_name,
                # "prompt_used": prompt
            }
            
        except Exception as e:
            logger.error(f"Error in prompt-based classification: {e}")
            return {
                "intent": "general_query",
                "confidence": 0.0,
                "entities": {},
                "original_text": text
            }

    def _analyze_text_with_rules(self, text: str) -> Dict[str, float]:
        """Analyze text using rule-based scoring for each category."""
        text_lower = text.lower()
        category_scores = {"Reminders": 0.0, "Ledger": 0.0, "Notes": 0.0, "Chitchat": 0.0}
        
        for category, rules in self.classification_rules.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in text_lower)
            score += keyword_matches * 0.4  # Increased weight for keywords
            
            # Check patterns
            pattern_matches = sum(1 for pattern in rules["patterns"] if re.search(pattern, text, re.IGNORECASE))
            score += pattern_matches * 0.7  # Increased weight for patterns
            
            # Check specific indicators
            if category == "Reminders":
                # Check for time indicators
                time_matches = sum(1 for indicator in rules["time_indicators"] if indicator in text_lower)
                score += time_matches * 0.5
                
            elif category == "Ledger": 
                # Check for money indicators
                money_matches = sum(1 for indicator in rules["money_indicators"] if indicator in text_lower)
                score += money_matches * 0.7
                # Check for dollar amounts
                if re.search(r'\$\d+|\b\d+\s*(dollars?|bucks?)\b', text, re.IGNORECASE):
                    score += 1.2
                # Extra boost for financial transaction patterns
                financial_patterns = [
                    r'\b[A-Z][a-z]+\s+(?:will\s+)?(?:give|pay)\s+(?:me|us)',
                    r'\b(?:give|pay)\s+(?:me|us)\s*\$',
                    r'\b[A-Z][a-z]+.*\$\d+',
                    r'\$\d+.*\b[A-Z][a-z]+',
                ]
                for pattern in financial_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        score += 1.5  # Strong boost for clear financial transactions
                        break
                    
            elif category == "Notes":
                # Check for content indicators
                content_matches = sum(1 for indicator in rules["content_indicators"] if indicator in text_lower)
                score += content_matches * 0.5
                # Reduce score if this looks like a financial transaction
                if re.search(r'\$\d+|\b\d+\s*(dollars?|bucks?)\b', text, re.IGNORECASE) and re.search(r'\b[A-Z][a-z]+\s+(?:will\s+)?(?:give|pay)', text, re.IGNORECASE):
                    score *= 0.3  # Heavily penalize note classification for financial transactions
                    
            elif category == "Chitchat":
                # Check for question indicators
                question_matches = sum(1 for indicator in rules["question_indicators"] if indicator in text_lower)
                score += question_matches * 0.4
            
            category_scores[category] = max(0.0, score)  # Ensure non-negative scores
        
        return category_scores

    def _normalize_text(self, text: str) -> str:
        """Normalize input text for better classification."""
        text = text.strip()
        # Convert multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        # Normalize time formats
        text = re.sub(r'(\d{1,2})\s*:\s*(\d{2})', r'\1:\2', text)
        # Normalize AM/PM formats
        text = re.sub(r'([ap])\.m\.', r'\1m', text, flags=re.IGNORECASE)
        return text

    def _create_fallback_response(self, text: str, multi_intent: bool) -> Dict[str, any]:
        """Create fallback response for error cases."""
        if multi_intent:
            return {
                "intents": [{
                    "type": "general_query",
                    "confidence": 0.0,
                    "entities": {},
                    "text_segment": text
                }],
                "original_text": text
            }
        else:
            return {
                "intent": "general_query",
                "confidence": 0.0,
                "entities": {},
                "original_text": text
            }
    
    async def _classify_multi_intent(self, text: str) -> Dict[str, any]:
        """
        Enhanced multi-intent detection and classification using prompt engineering.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with array of intents
        """
        logger.info(f"Multi-intent classification for: '{text}'")
        
        # Split text into segments using enhanced patterns
        segments = self._segment_text_for_multi_intent(text)
        
        # Process each segment
        intents = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            logger.info(f"Processing segment: '{segment}'")
            
            # Classify the segment using prompt engineering
            try:
                # Create prompt for this segment
                # prompt = f"Classify the following text into one of these categories: Reminders, Ledger, Notes, or Chitchat.\n\nText: {segment}\nCategory:"
                
                # Analyze segment using rules
                category_scores = self._analyze_text_with_rules(segment)
                
                # Get the best category
                best_category = max(category_scores.items(), key=lambda x: x[1])
                category_name = best_category[0]
                confidence = best_category[1]
                
                # Map category to intent
                intent_type = self.category_mapping.get(category_name, "general_query")
                
                # Extract entities for this segment
                entities = self._extract_entities(segment)
                
                # Only include if confidence is above threshold
                if confidence > 0.2:  # Lower threshold for multi-intent
                    intents.append({
                        "type": intent_type,
                        "confidence": confidence,
                        "entities": entities,
                        "text_segment": segment,
                        "category": category_name,
                        # "prompt_used": prompt
                    })
                    logger.info(f"Segment classified: {category_name} -> {intent_type} (confidence: {confidence:.3f})")
                else:
                    logger.warning(f"Segment confidence too low: {confidence:.3f} for '{segment}'")
                    
            except Exception as e:
                logger.error(f"Error processing segment '{segment}': {e}")
                continue
        
        logger.info(f"Multi-intent result: {len(intents)} intents detected")
        
        return {
            "intents": intents,
            "original_text": text
        }

    def _segment_text_for_multi_intent(self, text: str) -> List[str]:
        """
        Enhanced text segmentation for multiple intents.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of text segments, each potentially containing a separate intent
        """
        # First check if this looks like a single intent that shouldn't be split
        text_lower = text.lower()
        
        # Don't split if it's clearly a single coherent intent
        single_intent_patterns = [
            r'\b(add|create|make)\s+(a\s+)?(note|reminder)\s+to\b',  # "add a note to", "create reminder to"
            r'\b(note|write|jot)\s+(down\s+)?(that\s+)?',  # "note that", "write down"
            r'\b(remind\s+me\s+to|set\s+(a\s+)?reminder)\b',  # "remind me to", "set reminder"
            r'\b(john|sarah|mike|alex|david|mary)\s+(owes?|owed?|borrowed?|lent|paid?)\b',  # "John owes"
            r'\b(track|record|log)\s+(expense|payment|cost)\b',  # "track expense"
        ]
        
        # Only apply single intent check for shorter texts
        if len(text) < 100:
            for pattern in single_intent_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.info(f"Text '{text}' matches single intent pattern: {pattern}")
                    return [text]  # Don't split
        
        # Enhanced segmentation using multiple strategies
        segments = []
        
        # Strategy 1: Split by clear intent markers with 'and'
        intent_transition_patterns = [
            r'\s+and\s+set\s+(?:a\s+)?(?:reminder|note)',           # " and set a reminder", " and set note"
            r'\s+and\s+(?:remind|note)\s+',                         # " and remind ", " and note "
            r'\s+and\s+(?:I\s+want|I\s+need)',                     # " and I want", " and I need"
            r'\s+and\s+[A-Z][a-z]+\s+(?:will\s+)?(?:give|pay|owe)', # " and John will give", " and John owes"
            r'\s+and\s+(?:remind\s+me\s+to)',                      # " and remind me to"
        ]
        
        # Find split points using transition patterns
        split_points = []
        for pattern in intent_transition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                split_points.append(match.start())
        
        # Remove duplicates and sort
        split_points = sorted(list(set(split_points)))
        
        # Create segments based on split points
        if split_points:
            current_start = 0
            for split_point in split_points:
                if current_start < split_point:
                    segment = text[current_start:split_point].strip()
                    if len(segment) > 10:  # Minimum segment length
                        segments.append(segment)
                current_start = split_point
            
            # Add the last segment
            last_segment = text[current_start:].strip()
            if last_segment:
                # Clean up "and" at the beginning
                last_segment = re.sub(r'^and\s+', '', last_segment, flags=re.IGNORECASE)
                if len(last_segment) > 10:
                    segments.append(last_segment)
        
        # Strategy 2: If no clear splits found, try word-by-word analysis
        if len(segments) <= 1:
            segments = []
            current_segment = ""
            words = text.split()
            i = 0
            
            while i < len(words):
                word = words[i].lower()
                
                # Check for intent transition markers
                is_transition = False
                
                # Look for transition patterns
                if (word == 'and' and i + 1 < len(words)):
                    next_word = words[i + 1].lower()
                    
                    # Strong intent starters after 'and'
                    strong_starters = ['set', 'remind', 'note', 'john', 'sarah', 'mike', 'alex', 'david', 'mary', 'i']
                    
                    if next_word in strong_starters and len(current_segment.split()) >= 4:
                        is_transition = True
                        # Look ahead to see if this forms a valid intent
                        lookahead = ' '.join(words[i+1:i+5])  # Next 4 words
                        intent_indicators = ['set a', 'remind me', 'note to', 'will give', 'want to', 'owes me']
                        
                        if any(indicator in lookahead.lower() for indicator in intent_indicators):
                            is_transition = True
                
                if is_transition and current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
                    # Skip the transition word 'and'
                    i += 1
                    continue
                
                current_segment += " " + words[i]
                i += 1
            
            # Add the last segment
            if current_segment.strip():
                segments.append(current_segment.strip())
        
        # Strategy 3: Post-process segments to ensure quality
        cleaned_segments = []
        for segment in segments:
            segment = segment.strip()
            
            # Remove leading/trailing punctuation and conjunctions
            segment = re.sub(r'^[,.\s]*(?:and\s+)?', '', segment, flags=re.IGNORECASE)
            segment = re.sub(r'[,.\s]*$', '', segment)
            
            # Skip very short segments
            if len(segment) < 8:
                continue
            
            # Skip segments that don't contain meaningful intent words
            intent_words = ['remind', 'set', 'note', 'buy', 'owe', 'owes', 'give', 'pay', 'want', 'need', 'book', 'call']
            if not any(word in segment.lower() for word in intent_words):
                continue
                
            cleaned_segments.append(segment)
        
        # If we still don't have good segments, return original text
        if len(cleaned_segments) <= 1:
            cleaned_segments = [text]
        
        logger.info(f"Text segmentation: '{text}' -> {len(cleaned_segments)} segments: {cleaned_segments}")
        return cleaned_segments
    
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
                entities["time"] = [" ".join(str(x) for x in matches[0] if x)]
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
                entities["date"] = [matches[0] if isinstance(matches[0], str) else " ".join(matches[0])]
                break
        
        # Enhanced name extraction with priority for ledger contexts
        person_found = None
        
        # High priority: Direct ledger patterns (most accurate for financial contexts)
        ledger_name_patterns = [
            r'\b([A-Z][a-z]+)\s+(?:owes?|owed?)\s+(?:me|us)',          # "John owes me"
            r'\b([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay)\s+(?:me|us)', # "John will give me", "John pay me"
            r'(?:and\s+)?([A-Z][a-z]+)\s+(?:will\s+)?(?:give|pay)',    # "and John will give"
            r'\b([A-Z][a-z]+)\s+borrowed',                             # "John borrowed"
            r'borrowed\s+from\s+([A-Z][a-z]+)',                       # "borrowed from John"
            r'lent\s+(?:to\s+)?([A-Z][a-z]+)',                        # "lent to John"
            r'\b([A-Z][a-z]+)\s+lent',                                 # "John lent"
        ]
        
        for pattern in ledger_name_patterns:
            matches = re.findall(pattern, text)
            if matches:
                person_found = matches[0]
                break
        
        # Medium priority: General conversation patterns (if no ledger pattern found)
        if not person_found:
            general_name_patterns = [
                r'\bcall\s+([A-Z][a-z]+)\b',                          # "call John"
                r'\bmeet\s+(?:with\s+)?([A-Z][a-z]+)\b',              # "meet John", "meet with John"
                r'\bwith\s+([A-Z][a-z]+)\b',                          # "with John"
                r'\babout\s+([A-Z][a-z]+)\b',                         # "about John"
                r'\bto\s+([A-Z][a-z]+)\b',                            # "to John"
                r'\bfrom\s+([A-Z][a-z]+)\b',                          # "from John"
            ]
            
            for pattern in general_name_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    person_found = matches[0]
                    break
        
        # Low priority: Standalone capitalized words (only if no other pattern found)
        if not person_found:
            # Look for capitalized words that could be names, but exclude common words
            excluded_words = {
                'Me', 'You', 'I', 'He', 'She', 'They', 'That', 'Note', 'Money', 'And', 'Set', 
                'Will', 'Give', 'Pay', 'Owe', 'Owes', 'Owed', 'Buy', 'Get', 'Go', 'Come',
                'Make', 'Take', 'Put', 'Call', 'Tell', 'Ask', 'Say', 'See', 'Know', 'Think',
                'Want', 'Need', 'Have', 'Get', 'Do', 'Can', 'Will', 'Should', 'Would', 'Could',
                'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'All', 'Some', 'Any',
                'Remind', 'Reminder', 'Note', 'Ledger', 'Home', 'Lahore', 'Ticket', 'Book',
                'Chocolates'
            }
            
            words = text.split()
            for word in words:
                # Look for capitalized words that could be names
                clean_word = word.strip('.,!?";:')
                if (clean_word and clean_word[0].isupper() and len(clean_word) > 2 and 
                    clean_word not in excluded_words and
                    not clean_word.lower().startswith(('$', '€', '£', '¥')) and
                    not clean_word.isdigit()):
                    person_found = clean_word
                    break
        
        if person_found:
            entities["person"] = [person_found]
        
        # Extract monetary amounts for ledger entries
        money_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',                        # $50, $1,000, $50.00
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',               # 50 dollars, 1000 dollars
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*bucks?',                 # 50 bucks
        ]
        
        for pattern in money_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                entities["amount"] = [matches[0]]
                break
        
        # Extract content for notes (when note-related keywords are present)
        if any(word in text_lower for word in ['note', 'remember', 'write', 'jot', 'buy']):
            content_patterns = [
                r'note\s+(?:to\s+|about\s+|that\s+)?(.+)',
                r'remember\s+(?:to\s+|that\s+)?(.+)',
                r'write\s+(?:down\s+)?(.+)',
                r'jot\s+(?:down\s+)?(.+)',
                r'buy\s+(.+)',
            ]
            
            for pattern in content_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    # Clean up the content by removing trailing parts that don't belong to the note
                    content = re.sub(r'\s+and\s+(john|sarah|mike|alex|david|mary|remind|set|call).*$', '', content, flags=re.IGNORECASE)
                    entities["content"] = content
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

    async def process_audio_transcript(self, transcript: str) -> Dict[str, any]:
        """
        Process transcribed audio to determine intent(s) with multi-intent support.
        
        Args:
            transcript: Transcribed text from audio
            
        Returns:
            Dictionary containing intent classification results (multi-intent format)
        """
        try:
            # Use multi-intent classification to detect all intents
            result = await self.classify_intent(transcript, multi_intent=True)
            
            # Check if we have multiple intents
            if "intents" in result and isinstance(result["intents"], list):
                # Return multi-intent format
                return result
            else:
                # Convert single intent to multi-intent format for consistency
                single_intent = result.get("intent", "general_query")
                confidence = result.get("confidence", 0.0)
                entities = result.get("entities", {})
                
                return {
                    "intents": [{
                        "type": single_intent,
                        "confidence": confidence,
                        "entities": entities,
                        "text_segment": transcript
                    }],
                    "original_text": transcript
                }
                
        except Exception as e:
            logger.error(f"Error processing audio transcript: {e}")
            return {
                "intents": [{
                    "type": "general_query",
                    "confidence": 0.0,
                    "entities": {},
                    "text_segment": transcript
                }],
                "original_text": transcript
            }

    def _add_context_hints(self, text: str) -> str:
        """Add contextual hints to improve classification accuracy."""
        text_lower = text.lower()
        hints = []
        
        # Time-related hints for reminders
        if any(word in text_lower for word in ['tomorrow', 'today', 'am', 'pm', ':']) or \
           re.search(r'\d{1,2}(?::\d{2})?(?:\s*[ap]m)?', text_lower):
            hints.append("time_context")
        
        # Money-related hints for ledger
        if '$' in text or any(word in text_lower for word in ['dollars', 'bucks', 'paid', 'owe']):
            hints.append("money_context")
        
        # Note-taking hints
        if any(word in text_lower for word in ['write', 'note', 'save', 'jot']):
            hints.append("note_context")
        
        # Add hints as subtle markers in the text
        if hints:
            return f"{text} [{' '.join(hints)}]"
        return text
 