"""
Intent Processor API - Endpoints for processing intent classification results
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from firebase_auth import verify_firebase_token
from services.intent_processor_service import IntentProcessorService
from utils.logger import logger
from core.dependencies import get_intent_service

router = APIRouter()

# Pydantic models for request/response
class IntentData(BaseModel):
    """Model for intent classification data."""
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    original_text: str = Field(..., description="Original user input text")

class ProcessIntentRequest(BaseModel):
    """Request model for processing intent."""
    intent_data: IntentData
    
class ProcessIntentResponse(BaseModel):
    """Response model for processed intent."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    intent: str

class IntentProcessRequest(BaseModel):
    """Request model for intent processing."""
    intent_data: Dict[str, Any]

class MultiIntentTestRequest(BaseModel):
    """Request model for testing multi-intent processing."""
    text: str
    multi_intent: bool = True

class IntentProcessResponse(BaseModel):
    """Response model for intent processing."""
    success: bool
    message: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    total_intents: Optional[int] = None
    successful_intents: Optional[int] = None
    original_text: Optional[str] = None
    error: Optional[str] = None

# Initialize the service
intent_processor = IntentProcessorService()

@router.post("/process", response_model=IntentProcessResponse)
async def process_intent(
    request: IntentProcessRequest,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Process intent classification result and save to database.
    Supports both single intent and multi-intent processing.
    
    Expected input formats:
    
    Single intent:
    {
      "intent_data": {
        "intent": "create_reminder",
        "confidence": 0.95,
        "entities": {"time": "5 PM", "person": "John"},
        "original_text": "remind me to call John at 5 PM"
      }
    }
    
    Multi-intent:
    {
      "intent_data": {
        "intents": [
          {
            "type": "create_reminder",
            "confidence": 0.93,
            "entities": {"time": "01:00", "title": "sleep"},
            "text_segment": "Set a reminder for 1 a.m. to sleep"
          },
          {
            "type": "create_note",
            "confidence": 0.88,
            "entities": {"content": "buy chocolate"},
            "text_segment": "set a note to buy chocolate"
          }
        ],
        "original_text": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate"
      }
    }
    """
    try:
        current_user_id = current_user["uid"]
        
        result = await intent_processor.process_intent(
            intent_data=request.intent_data,
            user_id=current_user_id
        )
        
        return IntentProcessResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in intent processing endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process intent: {str(e)}"
        )

@router.post("/test-multi-intent")
async def test_multi_intent_processing(
    request: MultiIntentTestRequest,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Test endpoint for multi-intent processing.
    Takes raw text, classifies intents, and processes them.
    
    Example:
    {
      "text": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate",
      "multi_intent": true
    }
    """
    try:
        current_user_id = current_user["uid"]
        
        # Get intent service
        intent_service = get_intent_service()
        if not intent_service:
            raise HTTPException(
                status_code=503,
                detail="Intent classification service not available"
            )
        
        # Classify intent(s)
        logger.info(f"Testing multi-intent processing for text: '{request.text}'")
        intent_result = await intent_service.classify_intent(
            request.text, 
            multi_intent=request.multi_intent
        )
        
        # Process intent(s)
        processing_result = await intent_processor.process_intent(
            intent_data=intent_result,
            user_id=current_user_id
        )
        
        # Return combined result
        return {
            "success": True,
            "text": request.text,
            "multi_intent": request.multi_intent,
            "intent_classification": intent_result,
            "processing_result": processing_result
        }
        
    except Exception as e:
        logger.error(f"Error in multi-intent test endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test multi-intent processing: {str(e)}"
        )

@router.get("/supported-intents")
async def get_supported_intents(current_user: dict = Depends(verify_firebase_token)):
    """Get list of supported intents and their handlers."""
    try:
        return {
            "supported_intents": list(intent_processor.intent_handlers.keys()),
            "handler_mapping": {
                intent: handler.__name__ for intent, handler in intent_processor.intent_handlers.items()
            },
            "database_tables": {
                "create_reminder": "reminders",
                "create_note": "notes",
                "create_ledger": "ledger_entries",
                "add_expense": "ledger_entries",
                "chit_chat": "history_logs",
                "general_query": "history_logs"
            }
        }

    except Exception as e:
        logger.error(f"Error getting supported intents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported intents: {str(e)}"
        )

@router.get("/stats")
async def get_processing_stats(
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Get statistics about processed intents for the current user.
    """
    try:
        from connect_db import SessionLocal
        from models.models import Reminder, Note, LedgerEntry, HistoryLog
        from sqlalchemy import func
        
        user_id = current_user["uid"]
        db = SessionLocal()
        
        try:
            # Count records in each table for the user
            reminder_count = db.query(func.count(Reminder.id)).filter(Reminder.user_id == user_id).scalar()
            note_count = db.query(func.count(Note.id)).filter(Note.user_id == user_id).scalar()
            ledger_count = db.query(func.count(LedgerEntry.id)).filter(LedgerEntry.user_id == user_id).scalar()
            chat_count = db.query(func.count(HistoryLog.id)).filter(
                HistoryLog.user_id == user_id,
                HistoryLog.interaction_type == 'chit_chat'
            ).scalar()
            
            return {
                "user_id": user_id,
                "stats": {
                    "reminders_created": reminder_count or 0,
                    "notes_created": note_count or 0,
                    "ledger_entries_created": ledger_count or 0,
                    "chat_interactions": chat_count or 0,
                    "total_processed": (reminder_count or 0) + (note_count or 0) + (ledger_count or 0) + (chat_count or 0)
                }
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get processing stats: {str(e)}"
        ) 