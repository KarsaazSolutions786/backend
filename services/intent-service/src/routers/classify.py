from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import JSONResponse
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter

# Use secure shared authentication 
import sys
import os

try:
    from simple_auth import get_current_customer_id
except ImportError:
    # Fallback to local secure implementation
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi import Depends, HTTPException, status
    import jwt
    
    def get_current_customer_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> int:
        """Secure local implementation as fallback"""
        try:
            token = credentials.credentials
            secret_key = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
            
            # Validate production environment
            if os.getenv("ENVIRONMENT") == "production" and secret_key in [
                "your-secret-key-here", 
                "your-secret-key-here-change-in-production",
                "eindr-super-secret-key-change-in-production-123456789"
            ]:
                raise ValueError("Production environment requires a secure SECRET_KEY")
            
            payload = jwt.decode(
                token, 
                secret_key, 
                algorithms=["HS256"],
                options={"verify_signature": True, "verify_exp": True}
            )
            customer_id = payload.get("sub")
            if customer_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing subject"
                )
            return int(customer_id)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

router = APIRouter()
logger = logging.getLogger(__name__)

class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    customer_id: str
    context: Optional[Dict[str, Any]] = None

class IntentResponse(BaseModel):
    intent: str
    confidence: float
    entities: Dict[str, Any]
    suggested_action: Optional[str] = None

class TrainingData(BaseModel):
    text: str
    intent: str
    entities: Optional[Dict] = None

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.intent_examples = {
            "create_reminder": [
                "remind me to call the doctor tomorrow",
                "set an alarm for 8am",
                "don't forget to buy groceries",
                "schedule a meeting with John",
                "alert me when it's time to leave"
            ],
            "create_note": [
                "write down my shopping list",
                "save this information",
                "jot down the meeting notes",
                "record my thoughts",
                "take a note about the project"
            ],
            "add_expense": [
                "I spent $50 on groceries",
                "add $25 to my expenses",
                "I paid $100 for gas",
                "record a $15 lunch expense",
                "bought coffee for $5"
            ],
            "check_schedule": [
                "what's on my calendar today",
                "show my schedule",
                "what meetings do I have",
                "check my agenda",
                "what's planned for tomorrow"
            ],
            "get_reminders": [
                "show my reminders",
                "what alerts do I have",
                "list my notifications",
                "check my upcoming reminders",
                "what do I need to remember"
            ],
            "check_expenses": [
                "how much did I spend this month",
                "show my expense summary",
                "what's my spending report",
                "check my budget",
                "how much money did I use"
            ],
            "find_notes": [
                "find my shopping list",
                "search for meeting notes",
                "look for my project notes",
                "where are my notes",
                "find notes about the budget"
            ],
            "general_question": [
                "how are you",
                "what can you do",
                "help me with something",
                "what's the weather like",
                "tell me a joke"
            ]
        }
        self.intent_embeddings = {}
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
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
            
            # Pre-compute embeddings for intent examples
            self._compute_intent_embeddings()
            logger.info("Intent classifier initialized with sentence embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            self.model = None
    
    def _compute_intent_embeddings(self):
        """Pre-compute embeddings for all intent examples"""
        if not self.model:
            return
            
        for intent, examples in self.intent_examples.items():
            # Compute embeddings for all examples of this intent
            embeddings = self.model.encode(examples)
            # Store the mean embedding for this intent
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify intent using sentence embeddings"""
        text_lower = text.lower()
        
        # Fallback to keyword-based classification if model is not available
        if not self.model or not self.intent_embeddings:
            return self._keyword_classify(text)
        
        try:
            # Compute embedding for input text
            text_embedding = self.model.encode([text])[0]
            
            # Calculate similarities with all intent embeddings
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                similarity = cosine_similarity([text_embedding], [intent_embedding])[0][0]
                similarities[intent] = similarity
            
            # Find the best matching intent
            best_intent = max(similarities, key=similarities.get)
            best_confidence = similarities[best_intent]
            
            # If confidence is too low, fall back to keyword classification
            if best_confidence < 0.3:
                logger.info(f"Low embedding confidence ({best_confidence:.2f}), falling back to keywords")
                return self._keyword_classify(text)
            
        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}, falling back to keywords")
            return self._keyword_classify(text)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        return {
            "intent": best_intent,
            "confidence": min(best_confidence, 0.95),  # Cap at 95%
            "entities": entities,
            "suggested_action": self._get_action(best_intent)
        }
    
    def _keyword_classify(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        
        intent_patterns = {
            "create_reminder": ["remind", "reminder", "alert", "schedule", "don't forget", "set alarm"],
            "create_note": ["note", "write", "save", "jot", "record", "take note"],
            "add_expense": ["spent", "cost", "paid", "expense", "money", "bought", "dollar"],
            "check_schedule": ["schedule", "calendar", "agenda", "today", "tomorrow", "meeting"],
            "get_reminders": ["reminders", "alerts", "notifications", "upcoming", "what do I need"],
            "check_expenses": ["expenses", "spending", "money", "budget", "how much"],
            "find_notes": ["notes", "find", "search", "look for", "where are"],
            "general_question": ["how", "what", "when", "where", "why", "help", "tell me"]
        }
        
        best_intent = "general_question"
        best_confidence = 0.0
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = min(score / len(keywords) * 2, 0.95)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent
        
        entities = self._extract_entities(text)
        
        return {
            "intent": best_intent,
            "confidence": max(best_confidence, 0.1),  # Minimum confidence
            "entities": entities,
            "suggested_action": self._get_action(best_intent)
        }
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text using simple patterns"""
        entities = {}
        
        # Time patterns
        time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            r'\b(\d{1,2})\s*(am|pm)\b',
            r'\btomorrow\b',
            r'\btoday\b',
            r'\bnext\s+week\b',
            r'\bin\s+(\d+)\s+(minutes?|hours?|days?)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'time' not in entities:
                    entities['time'] = []
                entities['time'].append(match.group())
        
        # Money patterns
        money_patterns = [
            r'\$(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*dollars?',
            r'(\d+)\s*bucks?'
        ]
        
        for pattern in money_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'amount' not in entities:
                    entities['amount'] = []
                entities['amount'].append(match.group())
        
        # People patterns
        people_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+\b(?=\s+(?:said|told|mentioned|called))',  # Names before verbs
        ]
        
        for pattern in people_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if 'person' not in entities:
                    entities['person'] = []
                entities['person'].append(match.group())
        
        return entities
    
    def _get_action(self, intent: str) -> str:
        """Get suggested action for intent"""
        actions = {
            "create_reminder": "Create a new reminder with the specified time and description",
            "create_note": "Save the note content to your personal notes",
            "add_expense": "Record the expense in your expense tracker",
            "check_schedule": "Display your calendar for the requested time period",
            "get_reminders": "Show your active reminders and alerts",
            "check_expenses": "Generate expense report for the specified period",
            "find_notes": "Search your notes for the specified content",
            "general_question": "Provide a helpful response to the question"
        }
        return actions.get(intent, "Process the request")

# Global classifier instance
classifier = IntentClassifier()

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

@router.post("/classify", response_model=IntentResponse)
@limiter.limit("100/minute")
async def classify_intent(
    request: Request,
    request_data: IntentRequest, 
    current_customer_id: int = Depends(get_current_customer_id)
):
    """Classify intent from customer text"""
    try:
        # Classify the intent
        intent_result = classifier.classify(request_data.text)
        
        logger.info(f"Intent classified for user {current_customer_id}: '{request_data.text}' -> {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
        
        return IntentResponse(
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
            entities=intent_result.get("entities", {}),
            suggested_action=intent_result.get("suggested_action")
        )
        
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Intent classification failed"
        )

@router.get("/intents")
async def get_supported_intents():
    """Get list of supported intents"""
    return {
        "intents": list(classifier.intent_examples.keys()),
        "count": len(classifier.intent_examples)
    }

@router.post("/train")
async def train_model(training_data: List[TrainingData]):
    """Train the model with new examples (placeholder)"""
    # This would be implemented for dynamic training
    logger.info(f"Received {len(training_data)} training examples")
    return {"message": "Training data received", "count": len(training_data)}
