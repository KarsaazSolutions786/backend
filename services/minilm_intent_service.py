"""
MiniLM Intent Classification Service
Integrates fine-tuned MiniLM model for multi-intent classification with robust fallbacks.
"""

import os
import numpy as np
import pickle
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from core.config import settings
from utils.logger import logger

# Import PyTorch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch library available")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using fallback implementations")

# Import transformers with fallback
try:
    # Only import transformers if not in Railway/minimal mode
    is_minimal_mode = os.getenv("MINIMAL_MODE", "false").lower() == "true"
    is_railway_env = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    if TORCH_AVAILABLE and not (is_minimal_mode or is_railway_env):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F
        TRANSFORMERS_AVAILABLE = True
        logger.info("Transformers library available")
    else:
        TRANSFORMERS_AVAILABLE = False
        if is_minimal_mode or is_railway_env:
            logger.info("Railway/minimal mode detected - skipping transformers import")
        else:
            logger.warning("Transformers requires PyTorch - using rule-based classification")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("Transformers library not available. Install with: pip install transformers torch")

class MiniLMIntentService:
    """MiniLM-based Intent Classification service with multi-intent support and robust fallbacks."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Force CPU since PyTorch may not be available
        self.model_path = "models/Mini_LM.bin"
        self.model_name = "microsoft/DialoGPT-medium"  # Base model for tokenizer
        
        # Intent labels (must match training data)
        self.intent_labels = [
            "create_reminder",
            "create_note", 
            "create_ledger",
            "add_expense",
            "chit_chat",
            "general_query"
        ]
        
        # Enhanced rule-based classification patterns
        self.intent_patterns = {
            "create_reminder": [
                r'\b(remind|reminder|schedule|appointment|meeting|alarm|alert)\b',
                r'\b(at \d+|tomorrow|today|tonight|morning|evening|afternoon)\b',
                r'\b(call|contact|phone|message|text|email)\b.*\b(at|on|in)\b',
                r'\b(set|create|make|add).*\b(reminder|alarm|appointment)\b'
            ],
            "create_note": [
                r'\b(note|write|jot|remember|record|save|document)\b',
                r'\b(note about|write down|jot down|remember that|make a note)\b',
                r'\b(create|make|add|save).*\b(note|memo|list)\b',
                r'\b(brain|memory|recall|retain)\b',
                r'\b(buy|purchase|get)\s+(groceries|food|items?|chocolates)\b',  # Shopping related
                r'\bset\s+(?:a\s+)?note\s+to\s+buy\b',  # "set a note to buy"
            ],
            "create_ledger": [
                r'\b(owe|owes|owed|debt|loan|borrowed|lent|lend)\b',
                r'\$\d+|\b\d+\s*(dollar|buck|euro|pound)\b',
                r'\b(money|cash|payment|pay|paid|balance|due)\b',
                r'\b(track|record|log).*\b(money|expense|payment)\b',
                # Enhanced financial transaction patterns
                r'\b[A-Z][a-z]+\s+(?:will\s+)?(?:give|pay)\s+(?:me|us)',  # "John will give me"
                r'\b(?:give|pay)\s+(?:me|us)\s+\$\d+',                    # "give me $50"
                r'\b[A-Z][a-z]+.*\$\d+',                                  # "John ... $50"
                r'\$\d+.*\b[A-Z][a-z]+',                                  # "$50 ... John"
                r'\b(?:will\s+)?(?:give|pay).*\$\d+',                     # "will give ... $50"
                r'\b[A-Z][a-z]+\s+(?:owes?|owed?)\s+(?:me|us)',          # "John owes me"
                r'\b[A-Z][a-z]+\s+borrowed',                             # "John borrowed"
                r'borrowed\s+from\s+[A-Z][a-z]+',                        # "borrowed from John"
            ],
            "add_expense": [
                r'\b(spent|expense|cost|bought|purchase|paid for)\b',
                r'\b(shopping|grocery|store|mall|restaurant)\b',
                r'\b(bill|receipt|transaction|charge)\b',
                r'\$\d+|\b\d+\s*(dollar|buck|euro|pound)\b.*\b(spent|cost|paid)\b'
            ],
            "chit_chat": [
                r'\b(hello|hi|hey|good morning|good evening|good afternoon)\b',
                r'\b(how are you|what\'s up|how\'s it going|what\'s new)\b',
                r'\b(thank you|thanks|please|sorry|excuse me)\b',
                r'\b(help|assist|support|explain|tell me)\b'
            ]
        }
        
        # Multi-intent detection patterns (improved)
        self.multi_intent_separators = [
            r'\band\s+(?:also\s+)?(?:set|create|add|make|remind|schedule|note|record|track)',
            r'\balso\s+(?:please\s+)?(?:set|create|add|make|remind|schedule|note|record|track)',
            r'\bthen\s+(?:please\s+)?(?:set|create|add|make|remind|schedule|note|record|track)',
            r'\bplus\s+(?:please\s+)?(?:set|create|add|make|remind|schedule|note|record|track)',
            r'\band\s+(?:remind|note|track|record|schedule|log)',
            r'\band\s+(?:I|i)\s+(?:want|need|have to|should|would like to)',
            r'\band\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owes?|owed?|borrowed?|lent|paid?)',
            r'\band\s+(?:\$\d+|\d+\s*dollars?)',
            r'\;|\,\s+(?:and\s+)?(?:also|plus|then)',
            r'\.\s+(?:And|Also|Plus|Then)'
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned MiniLM model or initialize fallback."""
        try:
            if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
                logger.info(f"Loading MiniLM model on device: {self.device}")
                
                # Load tokenizer (using base model)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "microsoft/DialoGPT-medium",
                        trust_remote_code=True
                    )
                    logger.info("Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer: {e}")
                    return self._initialize_fallback_classification()
                
                # Load fine-tuned model
                if os.path.exists(self.model_path):
                    logger.info(f"Loading fine-tuned MiniLM model from {self.model_path}")
                    
                    try:
                        # Load model state from binary file
                        model_state = torch.load(self.model_path, map_location=self.device)
                        
                        # Initialize base model architecture
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            "microsoft/DialoGPT-medium",
                            num_labels=len(self.intent_labels),
                            trust_remote_code=True
                        )
                        
                        # Load fine-tuned weights
                        if isinstance(model_state, dict) and 'state_dict' in model_state:
                            self.model.load_state_dict(model_state['state_dict'])
                        else:
                            self.model.load_state_dict(model_state)
                        
                        self.model.to(self.device)
                        self.model.eval()
                        
                        logger.info("Fine-tuned MiniLM model loaded successfully")
                        return True
                        
                    except Exception as e:
                        logger.error(f"Failed to load fine-tuned model: {e}")
                        logger.info("Falling back to base model")
                        
                # Fallback to base model if fine-tuned model fails
                logger.info("Loading base MiniLM model")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/DialoGPT-medium",
                    num_labels=len(self.intent_labels),
                    trust_remote_code=True
                )
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("Base MiniLM model loaded successfully")
                return True
            else:
                logger.warning("PyTorch/Transformers not available - using enhanced rule-based classification")
                return self._initialize_fallback_classification()
                
        except Exception as e:
            logger.error(f"Failed to load MiniLM model: {e}")
            logger.info("Falling back to rule-based classification")
            return self._initialize_fallback_classification()
    
    def _initialize_fallback_classification(self):
        """Initialize enhanced rule-based classification."""
        try:
            logger.info("Enhanced rule-based classifier initialized with improved patterns")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize fallback classification: {e}")
            return False
    
    async def classify_intent(self, text: str, multi_intent: bool = True) -> Dict[str, Any]:
        """
        Classify intent(s) in the provided text.
        
        Args:
            text: Input text to classify
            multi_intent: Whether to detect multiple intents
            
        Returns:
            Dictionary containing classification results
        """
        try:
            # Clean and normalize text
            text = self._normalize_text(text)
            
            if multi_intent:
                return await self._classify_multi_intent(text)
            else:
                return await self._classify_single_intent(text)
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._create_fallback_response(text, multi_intent)
    
    async def _classify_single_intent(self, text: str) -> Dict[str, Any]:
        """Classify a single intent using MiniLM."""
        try:
            if not self.model or not self.tokenizer:
                return self._create_rule_based_classification(text, False)
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # Get predicted intent
            predicted_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_idx].item()
            intent = self.intent_labels[predicted_idx]
            
            # Post-process intent (map add_expense -> create_ledger)
            processed_intents = self._post_process_intents([intent])
            final_intent = processed_intents[0] if processed_intents else intent
            
            # Extract entities
            entities = self._extract_entities(text)
            
            logger.info(f"MiniLM classification: {final_intent} (confidence: {confidence:.3f})")
            
            return {
                "success": True,
                "intent": final_intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": text,
                "model_used": "minilm",
                "type": "single_intent"
            }
            
        except Exception as e:
            logger.error(f"MiniLM single intent classification failed: {e}")
            return self._create_rule_based_classification(text, False)
    
    async def _classify_multi_intent(self, text: str) -> Dict[str, Any]:
        """Classify multiple intents in text using MiniLM."""
        try:
            # First, segment the text for multi-intent detection
            segments = self._segment_text_for_multi_intent(text)
            
            if len(segments) <= 1:
                # Single intent, process normally
                return await self._classify_single_intent(text)
            
            # Process each segment
            results = []
            overall_confidence = 0.0
            
            for i, segment in enumerate(segments):
                segment_result = await self._classify_single_intent(segment)
                
                if segment_result.get("success", False):
                    results.append({
                        "intent": segment_result["intent"],
                        "confidence": segment_result["confidence"],
                        "entities": segment_result["entities"],
                        "segment": segment,
                        "segment_index": i
                    })
                    overall_confidence += segment_result["confidence"]
            
            if not results:
                return self._create_fallback_response(text, True)
            
            # Calculate overall confidence
            overall_confidence /= len(results)
            intents = [r["intent"] for r in results]
            
            # Post-process intents (map add_expense -> create_ledger, remove duplicates)
            processed_intents = self._post_process_intents(intents)
            
            logger.info(f"Multi-intent classification: {processed_intents} (overall confidence: {overall_confidence:.3f})")
            
            return {
                "success": True,
                "type": "multi_intent",
                "intents": processed_intents,
                "overall_confidence": overall_confidence,
                "segments": segments,
                "results": results,
                "original_text": text,
                "model_used": "minilm",
                "total_intents": len(processed_intents)
            }
            
        except Exception as e:
            logger.error(f"MiniLM multi-intent classification failed: {e}")
            return self._create_rule_based_classification(text, True)
    
    def _segment_text_for_multi_intent(self, text: str) -> List[str]:
        """Segment text into potential multi-intent components using enhanced logic."""
        try:
            # Normalize text
            text = text.strip()
            
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
            
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}")
            return [text]
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text using enhanced rule-based approach."""
        entities = {}
        
        try:
            # Keep original text for pattern matching (since it may have been normalized)
            original_text = text
            text_lower = text.lower()
            
            # Time entities
            time_patterns = [
                r'\b(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))\b',
                r'\b(\d{1,2}\s*(?:am|pm|AM|PM))\b',
                r'\b(tomorrow|today|tonight|morning|evening|afternoon)\b',
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(in\s+\d+\s+(?:minutes?|hours?|days?))\b',
                r'\b(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b'
            ]
            
            time_matches = []
            for pattern in time_patterns:
                matches = re.findall(pattern, original_text, re.IGNORECASE)
                time_matches.extend(matches)
            
            if time_matches:
                entities["time"] = time_matches
            
            # Enhanced person name extraction with priority for ledger contexts
            person_found = None
            
            # High priority: Direct ledger patterns (most accurate for financial contexts)
            # Use case-insensitive patterns since text might be normalized
            ledger_name_patterns = [
                r'\b(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\s+(?:owes?|owed?)\s+(?:me|us)',  # "john owes me"
                r'\b(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\s+(?:will\s+)?(?:give|pay)\s+(?:me|us)',  # "john will give me"
                r'(?:and\s+)?(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\s+(?:will\s+)?(?:give|pay)',  # "and john will give"
                r'\b(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\s+borrowed',  # "john borrowed"
                r'borrowed\s+from\s+(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)',  # "borrowed from john"
                r'lent\s+(?:to\s+)?(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)',  # "lent to john"
                r'\b(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\s+lent',  # "john lent"
            ]
            
            for pattern in ledger_name_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    person_found = matches[0].capitalize()  # Capitalize the first letter
                    break
            
            # Medium priority: General conversation patterns (if no ledger pattern found)
            if not person_found:
                general_name_patterns = [
                    r'\bcall\s+(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\b',  # "call john"
                    r'\bmeet\s+(?:with\s+)?(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\b',  # "meet john"
                    r'\bwith\s+(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\b',  # "with john"
                    r'\babout\s+(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\b',  # "about john"
                    r'\bto\s+(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\b',  # "to john"
                    r'\bfrom\s+(john|sarah|mike|alex|david|mary|jane|bob|alice|tom|lisa|jim|ana|sam|joe|amy)\b',  # "from john"
                    r'\b(mom|dad|mother|father|brother|sister|wife|husband)\b',  # Family
                    r'\b(doctor|dr\.?\s+\w+)\b'  # Professionals
                ]
                
                for pattern in general_name_patterns:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        person_found = matches[0].capitalize() if matches[0].lower() not in ['mom', 'dad', 'mother', 'father', 'brother', 'sister', 'wife', 'husband', 'doctor'] else matches[0]
                        break
            
            # Low priority: Try to find capitalized words in the original text if available
            if not person_found:
                # Look for capitalized words that could be names, but exclude common words
                excluded_words = {
                    'Me', 'You', 'I', 'He', 'She', 'They', 'That', 'Note', 'Money', 'And', 'Set', 
                    'Will', 'Give', 'Pay', 'Owe', 'Owes', 'Owed', 'Buy', 'Get', 'Go', 'Come',
                    'Make', 'Take', 'Put', 'Call', 'Tell', 'Ask', 'Say', 'See', 'Know', 'Think',
                    'Want', 'Need', 'Have', 'Get', 'Do', 'Can', 'Will', 'Should', 'Would', 'Could',
                    'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'All', 'Some', 'Any',
                    'Remind', 'Reminder', 'Note', 'Ledger', 'Home', 'Lahore', 'Ticket', 'Book',
                    'Chocolates', 'Today', 'Tomorrow', 'Tonight', 'Morning', 'Evening', 'Afternoon'
                }
                
                # Try to find the name in the original text before normalization
                words = original_text.split()
                for word in words:
                    # Look for capitalized words that could be names
                    clean_word = word.strip('.,!?";:')
                    if (clean_word and clean_word[0].isupper() and len(clean_word) > 2 and 
                        clean_word not in excluded_words and
                        not clean_word.lower().startswith(('$', '€', '£', '¥')) and
                        not clean_word.isdigit()):
                        person_found = clean_word
                        break
                
                # If still not found, try common names in lowercase text
                if not person_found:
                    common_names = ['john', 'sarah', 'mike', 'alex', 'david', 'mary', 'jane', 'bob', 'alice', 'tom', 'lisa', 'jim', 'ana', 'sam', 'joe', 'amy']
                    words_lower = text_lower.split()
                    for word in words_lower:
                        clean_word = word.strip('.,!?";:')
                        if clean_word in common_names:
                            person_found = clean_word.capitalize()
                            break
            
            if person_found:
                entities["person"] = [person_found]
            
            # Amount entities
            amount_patterns = [
                r'\$(\d+(?:\.\d{2})?)',
                r'\b(\d+(?:\.\d{2})?\s*dollars?)\b',
                r'\b(\d+\s*bucks?)\b'
            ]
            
            amount_matches = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, original_text, re.IGNORECASE)
                amount_matches.extend(matches)
            
            if amount_matches:
                entities["amount"] = amount_matches
            
            # Content entities (for notes)
            if any(word in text_lower for word in ['note', 'remember', 'write', 'jot', 'buy']):
                # Extract the main content after keywords
                content_patterns = [
                    r'note\s+(?:to\s+|about\s+|that\s+)?(.+)',
                    r'remember\s+(?:to\s+|that\s+)?(.+)',
                    r'write\s+(?:down\s+)?(.+)',
                    r'jot\s+(?:down\s+)?(.+)',
                    r'(?:to\s+)?buy\s+(.+)',                              # "buy chocolates" or "to buy chocolates"
                    r'^to\s+(.+)',                                        # segments starting with "to"
                ]
                
                for pattern in content_patterns:
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        # Clean up the content by removing trailing parts that don't belong to the note
                        content = re.sub(r'\s+and\s+(john|sarah|mike|alex|david|mary|remind|set|call).*$', '', content, flags=re.IGNORECASE)
                        
                        # For "to buy" patterns, include the action word for better context
                        if pattern == r'(?:to\s+)?buy\s+(.+)' or pattern == r'^to\s+(.+)':
                            # Check if this looks like a "buy" action
                            if 'buy' in original_text.lower():
                                # Extract the full "buy X" phrase
                                buy_match = re.search(r'(?:to\s+)?(buy\s+.+?)(?:\s+and|$)', original_text, re.IGNORECASE)
                                if buy_match:
                                    content = buy_match.group(1).strip()
                            elif pattern == r'^to\s+(.+)':
                                # For segments starting with "to", include the "to" if it makes sense
                                if original_text.lower().startswith('to ') and not any(skip_word in content.lower() for skip_word in ['and', 'then', 'also']):
                                    content = f"to {content}".strip()
                        
                        entities["content"] = content
                        break
                
                # If no pattern matched but we have "buy" in the text, try a more flexible approach
                if "content" not in entities and "buy" in text_lower:
                    # Extract everything after "buy" or "to buy"
                    buy_match = re.search(r'(?:to\s+)?buy\s+([^.]+?)(?:\s+and\s+\w+.*)?$', original_text, re.IGNORECASE)
                    if buy_match:
                        content = f"buy {buy_match.group(1).strip()}"
                        entities["content"] = content
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text for better classification."""
        try:
            # Basic cleaning
            text = text.strip().lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s\.,!?$]', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text normalization failed: {e}")
            return text
    
    def _create_rule_based_classification(self, text: str, multi_intent: bool) -> Dict[str, Any]:
        """Enhanced rule-based classification using improved patterns."""
        try:
            # Normalize text for better matching
            text_lower = self._normalize_text(text)
            
            # Calculate scores for each intent using pattern matching
            intent_scores = {}
            
            for intent, patterns in self.intent_patterns.items():
                score = 0
                matches = 0
                
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        matches += 1
                        # Weight patterns differently based on specificity
                        if len(pattern) > 50:  # More specific patterns get higher weight
                            score += 3
                        elif len(pattern) > 30:
                            score += 2
                        else:
                            score += 1
                
                # Normalize score by number of patterns
                if len(patterns) > 0:
                    intent_scores[intent] = (score / len(patterns)) * matches
                else:
                    intent_scores[intent] = 0
            
            # Handle cases where no patterns match
            if all(score == 0 for score in intent_scores.values()):
                # Fallback to simple keyword matching
                if any(word in text_lower for word in ['remind', 'reminder', 'schedule', 'alarm']):
                    intent_scores["create_reminder"] = 0.5
                elif any(word in text_lower for word in ['note', 'write', 'jot', 'remember']):
                    intent_scores["create_note"] = 0.5
                elif any(word in text_lower for word in ['owe', 'owes', 'money', '$']):
                    intent_scores["create_ledger"] = 0.5
                elif any(word in text_lower for word in ['spent', 'bought', 'cost', 'expense']):
                    intent_scores["add_expense"] = 0.5
                elif any(word in text_lower for word in ['hello', 'hi', 'help', 'thanks']):
                    intent_scores["chit_chat"] = 0.5
                else:
                    intent_scores["general_query"] = 0.3
            
            # Get the best matching intent(s)
            max_score = max(intent_scores.values()) if intent_scores else 0
            
            if max_score == 0:
                # Ultimate fallback
                best_intent = "general_query"
                confidence = 0.2
            else:
                best_intent = max(intent_scores, key=intent_scores.get)
                # Convert score to confidence (0-1 range)
                confidence = min(max_score / 3.0, 0.95)  # Cap at 95% confidence
            
            # Post-process the best intent
            processed_intents = self._post_process_intents([best_intent])
            final_intent = processed_intents[0] if processed_intents else best_intent
            
            # Extract entities
            entities = self._extract_entities(text)
            
            if multi_intent:
                # Check for multiple high-scoring intents
                high_scoring_intents = []
                threshold = max_score * 0.6  # At least 60% of max score
                
                for intent, score in intent_scores.items():
                    if score >= threshold and score > 0:
                        intent_confidence = min(score / 3.0, 0.95)
                        high_scoring_intents.append({
                            "intent": intent,
                            "confidence": intent_confidence,
                            "entities": entities,
                            "segment": text,
                            "segment_index": 0
                        })
                
                # If multiple intents found, return multi-intent result
                if len(high_scoring_intents) > 1:
                    overall_confidence = sum(r["confidence"] for r in high_scoring_intents) / len(high_scoring_intents)
                    
                    # Post-process intents (map add_expense -> create_ledger, remove duplicates)
                    raw_intents = [r["intent"] for r in high_scoring_intents]
                    processed_intents = self._post_process_intents(raw_intents)
                    
                    return {
                        "success": True,
                        "type": "multi_intent",
                        "intents": processed_intents,
                        "overall_confidence": overall_confidence,
                        "segments": [text],
                        "results": high_scoring_intents,
                        "original_text": text,
                        "model_used": "enhanced_rule_based",
                        "total_intents": len(processed_intents)
                    }
                elif len(high_scoring_intents) == 1:
                    # Single intent detected
                    result = high_scoring_intents[0]
                    
                    # Post-process the single intent
                    processed_intents = self._post_process_intents([result["intent"]])
                    final_intent = processed_intents[0] if processed_intents else result["intent"]
                    
                    return {
                        "success": True,
                        "type": "multi_intent",
                        "intents": processed_intents,
                        "overall_confidence": result["confidence"],
                        "segments": [text],
                        "results": high_scoring_intents,
                        "original_text": text,
                        "model_used": "enhanced_rule_based",
                        "total_intents": len(processed_intents)
                    }
            
            # Return single intent result
            return {
                "success": True,
                "intent": final_intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": text,
                "model_used": "enhanced_rule_based",
                "type": "single_intent",
                "pattern_scores": intent_scores
            }
                
        except Exception as e:
            logger.error(f"Enhanced rule-based classification failed: {e}")
            return self._create_fallback_response(text, multi_intent)
    
    def _create_fallback_response(self, text: str, multi_intent: bool) -> Dict[str, Any]:
        """Create fallback response when all classification methods fail."""
        # Default to general_query and post-process to ensure no add_expense
        default_intents = self._post_process_intents(["general_query"])
        
        if multi_intent:
            return {
                "success": False,
                "error": "Intent classification failed",
                "type": "multi_intent",
                "intents": default_intents,
                "overall_confidence": 0.1,
                "segments": [text],
                "results": [],
                "original_text": text,
                "model_used": "fallback"
            }
        else:
            return {
                "success": False,
                "error": "Intent classification failed",
                "intent": default_intents[0],
                "confidence": 0.1,
                "entities": {},
                "original_text": text,
                "model_used": "fallback",
                "type": "single_intent"
            }
    
    def is_ready(self) -> bool:
        """Check if the MiniLM service is ready."""
        return (self.model is not None and self.tokenizer is not None and TRANSFORMERS_AVAILABLE)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_available": self.model is not None,
            "tokenizer_available": self.tokenizer is not None,
            "model_path": self.model_path,
            "device": self.device,
            "intent_labels": self.intent_labels,
            "library": "transformers" if TRANSFORMERS_AVAILABLE else "not_available",
            "model_type": "fine_tuned_minilm" if os.path.exists(self.model_path) else "base_model",
            "supports_multi_intent": True
        }

    def _post_process_intents(self, intents: List[str]) -> List[str]:
        """
        Post-process intents to ensure compliance with business rules:
        1. Map 'add_expense' to 'create_ledger'
        2. Remove duplicates while preserving order
        
        Args:
            intents: List of intent labels
            
        Returns:
            Processed list of intents
        """
        # Map add_expense to create_ledger
        processed_intents = [intent.replace("add_expense", "create_ledger") for intent in intents]
        
        # Remove duplicates while preserving order
        deduplicated_intents = list(dict.fromkeys(processed_intents))
        
        return deduplicated_intents 