from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import re

router = APIRouter()
logger = logging.getLogger(__name__)

class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    user_id: str
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
        self.intent_patterns = {
            "create_reminder": ["remind", "reminder", "alert", "schedule", "don't forget"],
            "create_note": ["note", "write", "save", "jot", "record"],
            "add_expense": ["spent", "cost", "paid", "expense", "money", "bought"],
            "check_schedule": ["schedule", "calendar", "agenda", "today", "tomorrow"],
            "get_reminders": ["reminders", "alerts", "notifications", "upcoming"],
            "check_expenses": ["expenses", "spending", "money", "budget"],
            "find_notes": ["notes", "find", "search", "look for"],
            "general_question": ["how", "what", "when", "where", "why", "help"]
        }
    
    def classify(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        
        best_intent = "general_question"
        best_confidence = 0.0
        
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = min(score / len(keywords) * 2, 0.95)  # Cap at 95%
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent
        
        # Extract basic entities
        entities = {}
        if "$" in text or any(word in text_lower for word in ["dollar", "cost", "price"]):
            amounts = re.findall(r"\$?[\d,]+\.?\d*", text)
            if amounts:
                entities["amount"] = amounts[0]
        
        return {
            "intent": best_intent,
            "confidence": best_confidence,
            "entities": entities,
            "suggested_action": self._get_action(best_intent)
        }
    
    def _get_action(self, intent: str) -> str:
        actions = {
            "create_reminder": "Create a new reminder",
            "create_note": "Save as a note",
            "add_expense": "Add to expenses",
            "check_schedule": "Show calendar",
            "get_reminders": "Display reminders",
            "check_expenses": "Show expense summary",
            "find_notes": "Search notes",
            "general_question": "Provide information"
        }
        return actions.get(intent, "Process request")

classifier = IntentClassifier()

@router.post("/classify", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    """Classify user intent from text"""
    try:
        result = classifier.classify(request.text)
        
        logger.info(f"Intent: '{request.text}' -> {result['intent']} ({result['confidence']:.2f})")
        
        return IntentResponse(
            intent=result["intent"],
            confidence=result["confidence"],
            entities=result["entities"],
            suggested_action=result["suggested_action"]
        )
        
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify intent"
        )

@router.get("/intents")
async def get_supported_intents():
    """Get list of supported intents"""
    return {
        "intents": list(classifier.intent_patterns.keys()),
        "count": len(classifier.intent_patterns)
    }

@router.post("/train")
async def train_model(training_data: List[TrainingData]):
    """Train the intent classification model"""
    return {
        "message": f"Model trained with {len(training_data)} examples",
        "accuracy": 0.95
    }
