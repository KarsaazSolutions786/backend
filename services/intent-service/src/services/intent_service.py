import os
import logging
import asyncio
import re
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)

class IntentService:
    def __init__(self):
        self.model = None
        self.intent_embeddings = {}
        self.intent_patterns = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        
        # Initialize intent patterns and training data
        self._init_intent_patterns()
        
    def _init_intent_patterns(self):
        """Initialize intent patterns and example phrases"""
        self.intent_patterns = {
            "create_reminder": {
                "keywords": ["remind", "reminder", "alert", "schedule", "don't forget", "set reminder"],
                "patterns": [
                    r"remind me to (.+)",
                    r"set (?:a )?reminder (?:to|for) (.+)",
                    r"don't forget (?:to|about) (.+)",
                    r"schedule (?:a )?reminder (?:to|for) (.+)"
                ],
                "examples": [
                    "remind me to call mom at 5pm",
                    "set a reminder to buy groceries tomorrow",
                    "don't forget about the meeting",
                    "schedule reminder for doctor appointment"
                ]
            },
            "create_note": {
                "keywords": ["note", "write", "save", "jot", "record", "note down"],
                "patterns": [
                    r"(?:take|make|write|save) (?:a )?note (?:about|that|saying) (.+)",
                    r"jot down (.+)",
                    r"record (?:that|this) (.+)"
                ],
                "examples": [
                    "take a note about the project requirements",
                    "save note that coffee shop closes at 8pm",
                    "jot down the meeting notes",
                    "record this important information"
                ]
            },
            "add_expense": {
                "keywords": ["spent", "cost", "paid", "expense", "money", "bought", "purchase"],
                "patterns": [
                    r"(?:i )?spent \$?(\d+\.?\d*) (?:on|for) (.+)",
                    r"(?:it )?cost(?:s)? \$?(\d+\.?\d*) (?:for|to) (.+)",
                    r"paid \$?(\d+\.?\d*) (?:for|to) (.+)",
                    r"bought (.+) for \$?(\d+\.?\d*)"
                ],
                "examples": [
                    "I spent $25 on lunch today",
                    "it cost $50 for the book",
                    "paid $100 for groceries",
                    "bought coffee for $4.50"
                ]
            },
            "check_schedule": {
                "keywords": ["schedule", "calendar", "agenda", "today", "tomorrow", "upcoming"],
                "patterns": [
                    r"what's (?:on my|my) (?:schedule|calendar|agenda) (?:for )?(.+)",
                    r"show me (?:my )?(?:schedule|calendar) (?:for )?(.+)",
                    r"what (?:do i have|am i doing) (.+)"
                ],
                "examples": [
                    "what's on my schedule today",
                    "show me my calendar for tomorrow",
                    "what do I have planned this week",
                    "what's my agenda for Monday"
                ]
            },
            "get_reminders": {
                "keywords": ["reminders", "alerts", "notifications", "upcoming", "pending"],
                "patterns": [
                    r"show (?:me )?(?:my )?reminders",
                    r"what reminders do i have",
                    r"list (?:my )?(?:upcoming )?reminders"
                ],
                "examples": [
                    "show me my reminders",
                    "what reminders do I have",
                    "list upcoming reminders",
                    "show pending alerts"
                ]
            },
            "check_expenses": {
                "keywords": ["expenses", "spending", "money", "budget", "total", "spent"],
                "patterns": [
                    r"how much (?:did i|have i) spent (?:on|this|today|yesterday)",
                    r"show (?:me )?(?:my )?expenses (?:for )?(.+)",
                    r"what's my (?:total )?spending (?:for )?(.+)"
                ],
                "examples": [
                    "how much did I spend today",
                    "show me my expenses this month",
                    "what's my total spending",
                    "show expenses for this week"
                ]
            },
            "find_notes": {
                "keywords": ["notes", "find", "search", "look for", "show notes"],
                "patterns": [
                    r"find (?:my )?notes? (?:about|on|containing) (.+)",
                    r"search (?:for )?notes? (?:about|containing) (.+)",
                    r"show (?:me )?notes? (?:about|on) (.+)"
                ],
                "examples": [
                    "find notes about the project",
                    "search for notes containing meeting",
                    "show me notes about recipes",
                    "look for notes on travel plans"
                ]
            },
            "chit_chat": {
                "keywords": ["hello", "hi", "hey", "how are you", "thank you", "thanks", "goodbye", "bye"],
                "patterns": [
                    r"(?:hello|hi|hey) ?(?:there)?",
                    r"how are you",
                    r"(?:thank you|thanks) ?(?:so much)?",
                    r"(?:goodbye|bye) ?(?:bye)?"
                ],
                "examples": [
                    "hello there",
                    "hi how are you",
                    "thanks for helping",
                    "goodbye"
                ]
            },
            "general_question": {
                "keywords": ["how", "what", "when", "where", "why", "help", "can you"],
                "patterns": [
                    r"(?:how|what|when|where|why) (.+)",
                    r"can you help (?:me )?(?:with )?(.+)",
                    r"i need help (?:with )?(.+)"
                ],
                "examples": [
                    "how do I create a reminder",
                    "what can you do",
                    "can you help me with my schedule",
                    "I need help with expenses"
                ]
            }
        }
    
    async def load_model(self):
        """Load MiniLM model and create intent embeddings"""
        try:
            # Use local models directory within the service
            service_models_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            minilm_model_path = os.path.join(service_models_path, "Mini_LM.bin")
            
            # Also check environment variable for flexibility
            env_model_path = os.getenv("MODEL_PATH")
            if env_model_path:
                alt_minilm_path = os.path.join(env_model_path, "Mini_LM.bin")
                if os.path.exists(alt_minilm_path):
                    minilm_model_path = alt_minilm_path
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                minilm_model_path
            )
            
            # Create intent embeddings
            await loop.run_in_executor(
                self.executor,
                self._create_intent_embeddings
            )
            
            self.model_loaded = True
            logger.info("Intent classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            self.model_loaded = False
    
    def _load_model_sync(self, custom_model_path: str):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Attempting to load MiniLM model from: {custom_model_path}")
            
            if os.path.exists(custom_model_path):
                # Check if it's a Git LFS pointer file
                file_size = os.path.getsize(custom_model_path)
                if file_size < 1000:  # Less than 1KB, likely a Git LFS pointer
                    with open(custom_model_path, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                            logger.warning("Found Git LFS pointer file for MiniLM model")
                            logger.info("Loading standard sentence transformer model...")
                        else:
                            logger.info("Found actual MiniLM model file")
                else:
                    logger.info(f"Found actual MiniLM model file ({file_size / (1024*1024):.1f}MB)")
                    # For actual custom model loading, you would implement specific loading logic here
                    
            # Load standard MiniLM model
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("MiniLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise
    
    def _create_intent_embeddings(self):
        """Create embeddings for intent examples"""
        try:
            if not self.model:
                raise Exception("Model not loaded")
                
            logger.info("Creating intent embeddings...")
            
            for intent, data in self.intent_patterns.items():
                examples = data.get("examples", [])
                if examples:
                    # Create embeddings for all examples
                    embeddings = self.model.encode(examples)
                    # Store average embedding for the intent
                    self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
                    
            logger.info(f"Created embeddings for {len(self.intent_embeddings)} intents")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def _calculate_pattern_confidence(self, text: str, intent_data: Dict) -> float:
        """Calculate confidence based on keyword and pattern matching"""
        text_lower = text.lower()
        
        # Check keywords
        keywords = intent_data.get("keywords", [])
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0
        
        # Check patterns
        patterns = intent_data.get("patterns", [])
        pattern_score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_score = 1.0
                break
        
        # Combine scores
        return (keyword_score * 0.3) + (pattern_score * 0.7)
    
    def _calculate_semantic_confidence(self, text: str) -> Dict[str, float]:
        """Calculate semantic similarity using embeddings"""
        if not self.model or not self.intent_embeddings:
            return {}
        
        try:
            # Get text embedding
            text_embedding = self.model.encode([text])
            
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(text_embedding, [intent_embedding])[0][0]
                similarities[intent] = max(0, similarity)  # Ensure non-negative
                
            return similarities
            
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return {}
    
    def _extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities based on intent patterns"""
        entities = {}
        
        intent_data = self.intent_patterns.get(intent, {})
        patterns = intent_data.get("patterns", [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if intent == "create_reminder":
                    entities["task"] = groups[0] if groups else text
                elif intent == "create_note":
                    entities["content"] = groups[0] if groups else text
                elif intent == "add_expense":
                    if len(groups) >= 2:
                        entities["amount"] = groups[0]
                        entities["description"] = groups[1]
                    elif groups:
                        entities["description"] = groups[0]
                elif intent == "find_notes":
                    entities["search_term"] = groups[0] if groups else text
                break
        
        # Extract time entities for time-sensitive intents
        if intent in ["create_reminder", "check_schedule"]:
            time_entities = self._extract_time_entities(text)
            entities.update(time_entities)
        
        # Extract money amounts for expense-related intents
        if intent in ["add_expense", "check_expenses"]:
            money_entities = self._extract_money_entities(text)
            entities.update(money_entities)
        
        return entities
    
    def _extract_time_entities(self, text: str) -> Dict[str, Any]:
        """Extract time-related entities"""
        entities = {}
        
        # Time patterns
        time_patterns = [
            r"(?:at )?(\d{1,2}):(\d{2})(?:\s*(am|pm))?",
            r"(?:at )?(\d{1,2})(?:\s*(am|pm))",
            r"(?:in )?(\d+) (?:minutes?|mins?)",
            r"(?:in )?(\d+) (?:hours?|hrs?)",
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["time"] = match.group(0)
                break
        
        # Date patterns
        date_patterns = [
            r"(today|tomorrow|yesterday)",
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"(?:next|this) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["date"] = match.group(0)
                break
        
        return entities
    
    def _extract_money_entities(self, text: str) -> Dict[str, Any]:
        """Extract money amounts"""
        entities = {}
        
        money_patterns = [
            r"\$(\d+\.?\d*)",
            r"(\d+\.?\d*) dollars?",
            r"(\d+\.?\d*) cents?"
        ]
        
        for pattern in money_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["amount"] = match.group(1) if "cents" in pattern else match.group(0)
                break
        
        return entities
    
    async def classify_intent(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Classify intent using both pattern matching and semantic similarity"""
        if not self.model_loaded:
            raise Exception("Intent model not loaded")
        
        try:
            # Run classification in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._classify_sync,
                text,
                context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            raise
    
    def _classify_sync(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous intent classification"""
        text_lower = text.lower().strip()
        
        # Calculate pattern-based confidence for each intent
        pattern_confidences = {}
        for intent, intent_data in self.intent_patterns.items():
            pattern_confidences[intent] = self._calculate_pattern_confidence(text, intent_data)
        
        # Calculate semantic similarity
        semantic_confidences = self._calculate_semantic_confidence(text)
        
        # Combine confidences
        combined_confidences = {}
        for intent in self.intent_patterns.keys():
            pattern_conf = pattern_confidences.get(intent, 0)
            semantic_conf = semantic_confidences.get(intent, 0)
            
            # Weighted combination: 40% pattern, 60% semantic
            combined_conf = (pattern_conf * 0.4) + (semantic_conf * 0.6)
            combined_confidences[intent] = combined_conf
        
        # Find best intent
        best_intent = max(combined_confidences.keys(), key=lambda x: combined_confidences[x])
        best_confidence = combined_confidences[best_intent]
        
        # Minimum confidence threshold
        if best_confidence < 0.2:
            best_intent = "general_question"
            best_confidence = 0.5
        
        # Extract entities
        entities = self._extract_entities(text, best_intent)
        
        return {
            "intent": best_intent,
            "confidence": min(best_confidence, 0.95),  # Cap at 95%
            "entities": entities,
            "all_confidences": combined_confidences,
            "pattern_confidences": pattern_confidences,
            "semantic_confidences": semantic_confidences
        }
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        return self.model_loaded and self.model is not None
    
    def get_supported_intents(self) -> List[str]:
        """Get list of supported intents"""
        return list(self.intent_patterns.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "supported_intents": len(self.intent_patterns),
            "intents": list(self.intent_patterns.keys()),
            "embedding_dimension": 384,  # MiniLM-L6 dimension
            "status": "loaded"
        }

# Global instance
_intent_service: Optional[IntentService] = None

def get_intent_service() -> Optional[IntentService]:
    """Get the global Intent service instance"""
    return _intent_service

def initialize_intent_service() -> IntentService:
    """Initialize the global Intent service"""
    global _intent_service
    _intent_service = IntentService()
    return _intent_service 
import logging
import asyncio
import re
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)

class IntentService:
    def __init__(self):
        self.model = None
        self.intent_embeddings = {}
        self.intent_patterns = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        
        # Initialize intent patterns and training data
        self._init_intent_patterns()
        
    def _init_intent_patterns(self):
        """Initialize intent patterns and example phrases"""
        self.intent_patterns = {
            "create_reminder": {
                "keywords": ["remind", "reminder", "alert", "schedule", "don't forget", "set reminder"],
                "patterns": [
                    r"remind me to (.+)",
                    r"set (?:a )?reminder (?:to|for) (.+)",
                    r"don't forget (?:to|about) (.+)",
                    r"schedule (?:a )?reminder (?:to|for) (.+)"
                ],
                "examples": [
                    "remind me to call mom at 5pm",
                    "set a reminder to buy groceries tomorrow",
                    "don't forget about the meeting",
                    "schedule reminder for doctor appointment"
                ]
            },
            "create_note": {
                "keywords": ["note", "write", "save", "jot", "record", "note down"],
                "patterns": [
                    r"(?:take|make|write|save) (?:a )?note (?:about|that|saying) (.+)",
                    r"jot down (.+)",
                    r"record (?:that|this) (.+)"
                ],
                "examples": [
                    "take a note about the project requirements",
                    "save note that coffee shop closes at 8pm",
                    "jot down the meeting notes",
                    "record this important information"
                ]
            },
            "add_expense": {
                "keywords": ["spent", "cost", "paid", "expense", "money", "bought", "purchase"],
                "patterns": [
                    r"(?:i )?spent \$?(\d+\.?\d*) (?:on|for) (.+)",
                    r"(?:it )?cost(?:s)? \$?(\d+\.?\d*) (?:for|to) (.+)",
                    r"paid \$?(\d+\.?\d*) (?:for|to) (.+)",
                    r"bought (.+) for \$?(\d+\.?\d*)"
                ],
                "examples": [
                    "I spent $25 on lunch today",
                    "it cost $50 for the book",
                    "paid $100 for groceries",
                    "bought coffee for $4.50"
                ]
            },
            "check_schedule": {
                "keywords": ["schedule", "calendar", "agenda", "today", "tomorrow", "upcoming"],
                "patterns": [
                    r"what's (?:on my|my) (?:schedule|calendar|agenda) (?:for )?(.+)",
                    r"show me (?:my )?(?:schedule|calendar) (?:for )?(.+)",
                    r"what (?:do i have|am i doing) (.+)"
                ],
                "examples": [
                    "what's on my schedule today",
                    "show me my calendar for tomorrow",
                    "what do I have planned this week",
                    "what's my agenda for Monday"
                ]
            },
            "get_reminders": {
                "keywords": ["reminders", "alerts", "notifications", "upcoming", "pending"],
                "patterns": [
                    r"show (?:me )?(?:my )?reminders",
                    r"what reminders do i have",
                    r"list (?:my )?(?:upcoming )?reminders"
                ],
                "examples": [
                    "show me my reminders",
                    "what reminders do I have",
                    "list upcoming reminders",
                    "show pending alerts"
                ]
            },
            "check_expenses": {
                "keywords": ["expenses", "spending", "money", "budget", "total", "spent"],
                "patterns": [
                    r"how much (?:did i|have i) spent (?:on|this|today|yesterday)",
                    r"show (?:me )?(?:my )?expenses (?:for )?(.+)",
                    r"what's my (?:total )?spending (?:for )?(.+)"
                ],
                "examples": [
                    "how much did I spend today",
                    "show me my expenses this month",
                    "what's my total spending",
                    "show expenses for this week"
                ]
            },
            "find_notes": {
                "keywords": ["notes", "find", "search", "look for", "show notes"],
                "patterns": [
                    r"find (?:my )?notes? (?:about|on|containing) (.+)",
                    r"search (?:for )?notes? (?:about|containing) (.+)",
                    r"show (?:me )?notes? (?:about|on) (.+)"
                ],
                "examples": [
                    "find notes about the project",
                    "search for notes containing meeting",
                    "show me notes about recipes",
                    "look for notes on travel plans"
                ]
            },
            "chit_chat": {
                "keywords": ["hello", "hi", "hey", "how are you", "thank you", "thanks", "goodbye", "bye"],
                "patterns": [
                    r"(?:hello|hi|hey) ?(?:there)?",
                    r"how are you",
                    r"(?:thank you|thanks) ?(?:so much)?",
                    r"(?:goodbye|bye) ?(?:bye)?"
                ],
                "examples": [
                    "hello there",
                    "hi how are you",
                    "thanks for helping",
                    "goodbye"
                ]
            },
            "general_question": {
                "keywords": ["how", "what", "when", "where", "why", "help", "can you"],
                "patterns": [
                    r"(?:how|what|when|where|why) (.+)",
                    r"can you help (?:me )?(?:with )?(.+)",
                    r"i need help (?:with )?(.+)"
                ],
                "examples": [
                    "how do I create a reminder",
                    "what can you do",
                    "can you help me with my schedule",
                    "I need help with expenses"
                ]
            }
        }
    
    async def load_model(self):
        """Load MiniLM model and create intent embeddings"""
        try:
            # Use local models directory within the service
            service_models_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            minilm_model_path = os.path.join(service_models_path, "Mini_LM.bin")
            
            # Also check environment variable for flexibility
            env_model_path = os.getenv("MODEL_PATH")
            if env_model_path:
                alt_minilm_path = os.path.join(env_model_path, "Mini_LM.bin")
                if os.path.exists(alt_minilm_path):
                    minilm_model_path = alt_minilm_path
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                minilm_model_path
            )
            
            # Create intent embeddings
            await loop.run_in_executor(
                self.executor,
                self._create_intent_embeddings
            )
            
            self.model_loaded = True
            logger.info("Intent classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            self.model_loaded = False
    
    def _load_model_sync(self, custom_model_path: str):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Attempting to load MiniLM model from: {custom_model_path}")
            
            if os.path.exists(custom_model_path):
                # Check if it's a Git LFS pointer file
                file_size = os.path.getsize(custom_model_path)
                if file_size < 1000:  # Less than 1KB, likely a Git LFS pointer
                    with open(custom_model_path, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                            logger.warning("Found Git LFS pointer file for MiniLM model")
                            logger.info("Loading standard sentence transformer model...")
                        else:
                            logger.info("Found actual MiniLM model file")
                else:
                    logger.info(f"Found actual MiniLM model file ({file_size / (1024*1024):.1f}MB)")
                    # For actual custom model loading, you would implement specific loading logic here
                    
            # Load standard MiniLM model
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("MiniLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise
    
    def _create_intent_embeddings(self):
        """Create embeddings for intent examples"""
        try:
            if not self.model:
                raise Exception("Model not loaded")
                
            logger.info("Creating intent embeddings...")
            
            for intent, data in self.intent_patterns.items():
                examples = data.get("examples", [])
                if examples:
                    # Create embeddings for all examples
                    embeddings = self.model.encode(examples)
                    # Store average embedding for the intent
                    self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
                    
            logger.info(f"Created embeddings for {len(self.intent_embeddings)} intents")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def _calculate_pattern_confidence(self, text: str, intent_data: Dict) -> float:
        """Calculate confidence based on keyword and pattern matching"""
        text_lower = text.lower()
        
        # Check keywords
        keywords = intent_data.get("keywords", [])
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0
        
        # Check patterns
        patterns = intent_data.get("patterns", [])
        pattern_score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_score = 1.0
                break
        
        # Combine scores
        return (keyword_score * 0.3) + (pattern_score * 0.7)
    
    def _calculate_semantic_confidence(self, text: str) -> Dict[str, float]:
        """Calculate semantic similarity using embeddings"""
        if not self.model or not self.intent_embeddings:
            return {}
        
        try:
            # Get text embedding
            text_embedding = self.model.encode([text])
            
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(text_embedding, [intent_embedding])[0][0]
                similarities[intent] = max(0, similarity)  # Ensure non-negative
                
            return similarities
            
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return {}
    
    def _extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities based on intent patterns"""
        entities = {}
        
        intent_data = self.intent_patterns.get(intent, {})
        patterns = intent_data.get("patterns", [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if intent == "create_reminder":
                    entities["task"] = groups[0] if groups else text
                elif intent == "create_note":
                    entities["content"] = groups[0] if groups else text
                elif intent == "add_expense":
                    if len(groups) >= 2:
                        entities["amount"] = groups[0]
                        entities["description"] = groups[1]
                    elif groups:
                        entities["description"] = groups[0]
                elif intent == "find_notes":
                    entities["search_term"] = groups[0] if groups else text
                break
        
        # Extract time entities for time-sensitive intents
        if intent in ["create_reminder", "check_schedule"]:
            time_entities = self._extract_time_entities(text)
            entities.update(time_entities)
        
        # Extract money amounts for expense-related intents
        if intent in ["add_expense", "check_expenses"]:
            money_entities = self._extract_money_entities(text)
            entities.update(money_entities)
        
        return entities
    
    def _extract_time_entities(self, text: str) -> Dict[str, Any]:
        """Extract time-related entities"""
        entities = {}
        
        # Time patterns
        time_patterns = [
            r"(?:at )?(\d{1,2}):(\d{2})(?:\s*(am|pm))?",
            r"(?:at )?(\d{1,2})(?:\s*(am|pm))",
            r"(?:in )?(\d+) (?:minutes?|mins?)",
            r"(?:in )?(\d+) (?:hours?|hrs?)",
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["time"] = match.group(0)
                break
        
        # Date patterns
        date_patterns = [
            r"(today|tomorrow|yesterday)",
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"(?:next|this) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["date"] = match.group(0)
                break
        
        return entities
    
    def _extract_money_entities(self, text: str) -> Dict[str, Any]:
        """Extract money amounts"""
        entities = {}
        
        money_patterns = [
            r"\$(\d+\.?\d*)",
            r"(\d+\.?\d*) dollars?",
            r"(\d+\.?\d*) cents?"
        ]
        
        for pattern in money_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["amount"] = match.group(1) if "cents" in pattern else match.group(0)
                break
        
        return entities
    
    async def classify_intent(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Classify intent using both pattern matching and semantic similarity"""
        if not self.model_loaded:
            raise Exception("Intent model not loaded")
        
        try:
            # Run classification in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._classify_sync,
                text,
                context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            raise
    
    def _classify_sync(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous intent classification"""
        text_lower = text.lower().strip()
        
        # Calculate pattern-based confidence for each intent
        pattern_confidences = {}
        for intent, intent_data in self.intent_patterns.items():
            pattern_confidences[intent] = self._calculate_pattern_confidence(text, intent_data)
        
        # Calculate semantic similarity
        semantic_confidences = self._calculate_semantic_confidence(text)
        
        # Combine confidences
        combined_confidences = {}
        for intent in self.intent_patterns.keys():
            pattern_conf = pattern_confidences.get(intent, 0)
            semantic_conf = semantic_confidences.get(intent, 0)
            
            # Weighted combination: 40% pattern, 60% semantic
            combined_conf = (pattern_conf * 0.4) + (semantic_conf * 0.6)
            combined_confidences[intent] = combined_conf
        
        # Find best intent
        best_intent = max(combined_confidences.keys(), key=lambda x: combined_confidences[x])
        best_confidence = combined_confidences[best_intent]
        
        # Minimum confidence threshold
        if best_confidence < 0.2:
            best_intent = "general_question"
            best_confidence = 0.5
        
        # Extract entities
        entities = self._extract_entities(text, best_intent)
        
        return {
            "intent": best_intent,
            "confidence": min(best_confidence, 0.95),  # Cap at 95%
            "entities": entities,
            "all_confidences": combined_confidences,
            "pattern_confidences": pattern_confidences,
            "semantic_confidences": semantic_confidences
        }
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        return self.model_loaded and self.model is not None
    
    def get_supported_intents(self) -> List[str]:
        """Get list of supported intents"""
        return list(self.intent_patterns.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "supported_intents": len(self.intent_patterns),
            "intents": list(self.intent_patterns.keys()),
            "embedding_dimension": 384,  # MiniLM-L6 dimension
            "status": "loaded"
        }

# Global instance
_intent_service: Optional[IntentService] = None

def get_intent_service() -> Optional[IntentService]:
    """Get the global Intent service instance"""
    return _intent_service

def initialize_intent_service() -> IntentService:
    """Initialize the global Intent service"""
    global _intent_service
    _intent_service = IntentService()
    return _intent_service 
 
 