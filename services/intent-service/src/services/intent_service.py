import os
import logging
import asyncio
import re
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class IntentService:
    def __init__(self):
        self.model = None
        self.intent_patterns = {}
        self.intent_embeddings = {}
        self.model_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize intent patterns
        self._init_intent_patterns()
        
    def _init_intent_patterns(self):
        """Initialize intent patterns and keywords"""
        self.intent_patterns = {
            "create_reminder": {
                "keywords": ["remind", "reminder", "alert", "schedule", "don't forget", "set alarm"],
                "patterns": [r"remind me to", r"set.*alarm", r"don't forget to", r"schedule.*for"],
                "examples": [
                    "remind me to call the doctor tomorrow",
                    "set an alarm for 8am",
                    "don't forget to buy groceries",
                    "schedule a meeting with John"
                ]
            },
            "create_note": {
                "keywords": ["note", "write", "save", "jot", "record", "take note"],
                "patterns": [r"write down", r"save.*information", r"jot down", r"take.*note"],
                "examples": [
                    "write down my shopping list",
                    "save this information",
                    "jot down the meeting notes",
                    "take a note about the project"
                ]
            },
            "add_expense": {
                "keywords": ["spent", "cost", "paid", "expense", "money", "bought", "dollar"],
                "patterns": [r"spent.*\$", r"paid.*\$", r"cost.*\$", r"bought.*\$"],
                "examples": [
                    "I spent $50 on groceries",
                    "add $25 to my expenses",
                    "I paid $100 for gas",
                    "bought coffee for $5"
                ]
            },
            "check_schedule": {
                "keywords": ["schedule", "calendar", "agenda", "today", "tomorrow", "meeting"],
                "patterns": [r"what.*schedule", r"show.*calendar", r"what.*meetings", r"check.*agenda"],
                "examples": [
                    "what's on my calendar today",
                    "show my schedule",
                    "what meetings do I have",
                    "check my agenda"
                ]
            },
            "get_reminders": {
                "keywords": ["reminders", "alerts", "notifications", "upcoming", "what do I need"],
                "patterns": [r"show.*reminders", r"what.*alerts", r"list.*notifications"],
                "examples": [
                    "show my reminders",
                    "what alerts do I have",
                    "list my notifications",
                    "check my upcoming reminders"
                ]
            },
            "check_expenses": {
                "keywords": ["expenses", "spending", "money", "budget", "how much"],
                "patterns": [r"how much.*spent", r"show.*expenses", r"check.*budget"],
                "examples": [
                    "how much did I spend this month",
                    "show my expense summary",
                    "check my budget",
                    "what's my spending report"
                ]
            },
            "find_notes": {
                "keywords": ["notes", "find", "search", "look for", "where are"],
                "patterns": [r"find.*notes", r"search.*for", r"where.*notes"],
                "examples": [
                    "find my shopping list",
                    "search for meeting notes",
                    "where are my notes",
                    "find notes about the budget"
                ]
            },
            "general_question": {
                "keywords": ["how", "what", "when", "where", "why", "help", "tell me"],
                "patterns": [r"how.*", r"what.*", r"when.*", r"where.*", r"why.*"],
                "examples": [
                    "how are you",
                    "what can you do",
                    "help me with something",
                    "tell me a joke"
                ]
            }
        }
    
    async def load_model(self):
        """Load the sentence transformer model"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                ""
            )
            self.model_loaded = True
            logger.info("Intent service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize intent service: {e}")
            self.model_loaded = False
    
    def _load_model_sync(self, custom_model_path: str):
        """Load the sentence transformer model synchronously"""
        try:
            # Determine the model path
            MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "../../../models"))
            MINILM_MODEL_PATH = os.path.join(MODEL_PATH, "all-MiniLM-L6-v2", "sentence-transformers_all-MiniLM-L6-v2")
            
            if os.path.exists(MINILM_MODEL_PATH):
                logger.info(f"Loading all-MiniLM-L6-v2 model from: {MINILM_MODEL_PATH}")
                self.model = SentenceTransformer(MINILM_MODEL_PATH)
            else:
                logger.warning(f"Local model not found at {MINILM_MODEL_PATH}, using online model")
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Create intent embeddings
            self._create_intent_embeddings()
            logger.info("Intent service model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            self.model = None
    
    def _create_intent_embeddings(self):
        """Create embeddings for all intent examples"""
        try:
            if not self.model:
                return
            
            for intent_name, intent_data in self.intent_patterns.items():
                examples = intent_data.get("examples", [])
                if examples:
                    # Compute embeddings for all examples of this intent
                    embeddings = self.model.encode(examples)
                    # Store average embedding for the intent
                    self.intent_embeddings[intent_name] = np.mean(embeddings, axis=0)
                    
            logger.info(f"Created embeddings for {len(self.intent_embeddings)} intents")
            
        except Exception as e:
            logger.error(f"Error creating intent embeddings: {e}")
    
    def _calculate_pattern_confidence(self, text: str, intent_data: Dict) -> float:
        """Calculate confidence based on pattern matching"""
        patterns = intent_data.get("patterns", [])
        keywords = intent_data.get("keywords", [])
        
        # Check keyword presence
        keyword_score = 0
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                keyword_score += 1
        
        keyword_confidence = keyword_score / len(keywords) if keywords else 0
        
        # Check pattern matching
        pattern_score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_score += 1
        
        pattern_confidence = pattern_score / len(patterns) if patterns else 0
        
        # Combine scores
        return (keyword_confidence * 0.4) + (pattern_confidence * 0.6)
    
    def _calculate_semantic_confidence(self, text: str) -> Dict[str, float]:
        """Calculate semantic similarity with intent embeddings"""
        try:
            if not self.model or not self.intent_embeddings:
                return {}
            
            # Encode input text
            text_embedding = self.model.encode([text])[0]
            
            # Calculate similarities with all intents
            similarities = {}
            for intent_name, intent_embedding in self.intent_embeddings.items():
                similarity = cosine_similarity(
                    [text_embedding], 
                    [intent_embedding]
                )[0][0]
                similarities[intent_name] = float(similarity)
                
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating semantic confidence: {e}")
            return {}
    
    def _extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities from text based on intent"""
        entities = {}
        
        if intent == "create_reminder":
            entities.update(self._extract_time_entities(text))
        elif intent == "add_expense":
            entities.update(self._extract_money_entities(text))
        
        return entities
    
    def _extract_time_entities(self, text: str) -> Dict[str, Any]:
        """Extract time-related entities"""
        entities = {}
        
        # Simple time patterns
        time_patterns = {
            "today": r"\b(today|tonight)\b",
            "tomorrow": r"\b(tomorrow)\b",
            "next_week": r"\b(next week)\b",
            "next_month": r"\b(next month)\b",
            "specific_time": r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b",
            "relative_time": r"\b(in \d+ (?:minutes?|hours?|days?|weeks?))\b"
        }
        
        for entity_type, pattern in time_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return entities
    
    def _extract_money_entities(self, text: str) -> Dict[str, Any]:
        """Extract money-related entities"""
        entities = {}
        
        # Money pattern
        money_pattern = r"\$?(\d+\.?\d*)"
        matches = re.findall(money_pattern, text)
        if matches:
            entities["amount"] = float(matches[0])
        
        return entities
    
    async def classify_intent(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Classify intent from text"""
        if not self.model_loaded:
            raise Exception("Intent model not loaded")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._classify_sync,
            text,
            context
        )
        
        return result
    
    def _classify_sync(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous intent classification"""
        try:
            # Calculate pattern-based confidence
            pattern_confidences = {}
            for intent_name, intent_data in self.intent_patterns.items():
                confidence = self._calculate_pattern_confidence(text, intent_data)
                pattern_confidences[intent_name] = confidence
            
            # Calculate semantic confidence
            semantic_confidences = self._calculate_semantic_confidence(text)
            
            # Combine confidences
            combined_confidences = {}
            for intent_name in self.intent_patterns.keys():
                pattern_conf = pattern_confidences.get(intent_name, 0)
                semantic_conf = semantic_confidences.get(intent_name, 0)
                # Weight pattern matching higher for specific intents
                combined_confidences[intent_name] = (pattern_conf * 0.7) + (semantic_conf * 0.3)
            
            # Find best intent
            best_intent = max(combined_confidences.items(), key=lambda x: x[1])
            intent_name, confidence = best_intent
            
            # Extract entities
            entities = self._extract_entities(text, intent_name)
            
            return {
                "intent": intent_name,
                "confidence": confidence,
                "entities": entities,
                "text": text,
                "pattern_confidence": pattern_confidences.get(intent_name, 0),
                "semantic_confidence": semantic_confidences.get(intent_name, 0),
                "all_confidences": combined_confidences
            }
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "text": text,
                "error": str(e)
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
            "model_name": "all-MiniLM-L6-v2",
            "parameters": "80M",
            "embedding_dimension": 384,
            "supported_intents": len(self.intent_patterns),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "status": "loaded",
            "model_location": "local"
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
 
 