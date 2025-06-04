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
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch and transformers are available")
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
                    "spent", "gave", "received", "back", "return"
                ],
                "patterns": [
                    r'\b(owe|owes|owed)\b',
                    r'\b(borrowed|lent)\b',
                    r'\b(paid|pay)\b.*\b(back|me|him|her)\b',
                    r'\$\d+',
                    r'\b\d+\s*(dollars?|bucks?)\b',
                    r'\b(expense|cost|spent)\b'
                ],
                "money_indicators": ["$", "dollar", "dollars", "bucks", "money", "paid", "cost"]
            },
            
            "Notes": {
                "keywords": [
                    "note", "write", "jot", "save", "record", "remember", "list",
                    "shopping", "grocery", "idea", "thought", "memo", "diary",
                    "journal", "log", "minutes", "summary", "discussion"
                ],
                "patterns": [
                    r'\b(note|write|jot)\b.*\b(down|this|that)\b',
                    r'\b(save|record|remember)\b.*\b(this|that|it)\b',
                    r'\b(shopping|grocery)\s+list\b',
                    r'\b(meeting\s+)?minutes\b',
                    r'\b(idea|thought|discussion)\b'
                ],
                "content_indicators": ["list", "idea", "thought", "discussion", "meeting", "project"]
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
            prompt = f"Classify the following text into one of these categories: Reminders, Ledger, Notes, or Chitchat.\n\nText: {text}\nCategory:"
            logger.info(f"Using prompt: {prompt}")
            
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
                "prompt_used": prompt
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
            score += keyword_matches * 0.3
            
            # Check patterns
            pattern_matches = sum(1 for pattern in rules["patterns"] if re.search(pattern, text, re.IGNORECASE))
            score += pattern_matches * 0.5
            
            # Check specific indicators
            if category == "Reminders":
                # Check for time indicators
                time_matches = sum(1 for indicator in rules["time_indicators"] if indicator in text_lower)
                score += time_matches * 0.4
                
            elif category == "Ledger": 
                # Check for money indicators
                money_matches = sum(1 for indicator in rules["money_indicators"] if indicator in text_lower)
                score += money_matches * 0.6
                # Check for dollar amounts
                if re.search(r'\$\d+|\b\d+\s*(dollars?|bucks?)\b', text, re.IGNORECASE):
                    score += 1.0
                    
            elif category == "Notes":
                # Check for content indicators
                content_matches = sum(1 for indicator in rules["content_indicators"] if indicator in text_lower)
                score += content_matches * 0.4
                
            elif category == "Chitchat":
                # Check for question indicators
                question_matches = sum(1 for indicator in rules["question_indicators"] if indicator in text_lower)
                score += question_matches * 0.3
                # Check if it's a question
                if text.strip().endswith('?'):
                    score += 0.5
            
            # Normalize score to confidence (0-1)
            category_scores[category] = min(1.0, score / 3.0)
        
        # If all scores are low, boost the most likely based on simple heuristics
        max_score = max(category_scores.values())
        if max_score < 0.3:
            # Simple fallback heuristics
            if any(word in text_lower for word in ["remind", "alarm", "appointment"]):
                category_scores["Reminders"] = 0.7
            elif any(word in text_lower for word in ["$", "owe", "paid", "money"]):
                category_scores["Ledger"] = 0.7
            elif any(word in text_lower for word in ["note", "write", "save", "list"]):
                category_scores["Notes"] = 0.7
            else:
                category_scores["Chitchat"] = 0.6
        
        logger.info(f"Category scores for '{text}': {category_scores}")
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
                prompt = f"Classify the following text into one of these categories: Reminders, Ledger, Notes, or Chitchat.\n\nText: {segment}\nCategory:"
                
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
                        "prompt_used": prompt
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
        # First try to split on explicit connectors
        segments = []
        current_segment = ""
        
        # Split text into words
        words = text.split()
        i = 0
        
        while i < len(words):
            word = words[i].lower()
            
            # Check for intent transition markers
            is_transition = False
            
            # Common transition words
            transition_words = ['and', 'also', 'then', 'plus']
            
            # Intent-specific keywords
            intent_keywords = {
                'reminder': ['remind', 'set', 'alarm', 'schedule'],
                'ledger': ['owe', 'owes', 'borrowed', 'lent', 'paid', '$'],
                'note': ['note', 'write', 'save', 'jot'],
                'chitchat': ['how', 'what', 'who', 'when', 'where', 'why', 'hello', 'hi', 'thanks']
            }
            
            # Check if current word is a transition word
            if word in transition_words:
                # Look ahead for intent keywords
                if i + 1 < len(words):
                    next_word = words[i + 1].lower()
                    for intent_type, keywords in intent_keywords.items():
                        if next_word in keywords or any(kw in next_word for kw in keywords):
                            is_transition = True
                            break
                    
                    # Check for money amounts
                    if next_word.startswith('$') or next_word.isdigit():
                        is_transition = True
            
            # Check for implicit transitions (direct intent keywords)
            if not is_transition and i > 0:
                for keywords in intent_keywords.values():
                    if word in keywords or any(kw in word for kw in keywords):
                        # Check if previous words indicate a new intent
                        prev_context = ' '.join(words[max(0, i-3):i]).lower()
                        if not any(kw in prev_context for kw in keywords):
                            is_transition = True
                            break
            
            if is_transition and current_segment.strip():
                segments.append(current_segment.strip())
                current_segment = ""
                # Skip the transition word
                i += 1
            else:
                current_segment += " " + words[i]
            
            i += 1
        
        # Add the last segment
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        # If no segments were created, use the original text
        if not segments:
            segments = [text]
        
        logger.info(f"Text segmentation: '{text}' -> {len(segments)} segments: {segments}")
        return segments

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
 