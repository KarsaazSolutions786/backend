"""
Intent Processor API - Endpoints for processing intent classification results
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from firebase_auth import verify_firebase_token
from services.intent_processor_service import IntentProcessorService
from utils.logger import logger

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

# Initialize the service
intent_processor = IntentProcessorService()

@router.post("/process", response_model=ProcessIntentResponse)
async def process_intent(
    request: ProcessIntentRequest,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Process intent classification result and save to appropriate database table.
    
    This endpoint takes intent classification results and:
    - For create_reminder: saves to reminders table with extracted time and person
    - For create_note: saves to notes table
    - For create_ledger/add_expense: saves to ledger_entries table with amount and contact
    - For chit_chat: saves to history_logs table
    """
    try:
        user_id = current_user["uid"]
        
        # Process the intent using the service
        result = await intent_processor.process_intent(
            intent_data=request.intent_data.dict(),
            user_id=user_id
        )
        
        logger.info(f"Intent processed for user {user_id}: {result['intent']}")
        
        return ProcessIntentResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in process_intent endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process intent: {str(e)}"
        )

@router.post("/process-batch")
async def process_batch_intents(
    intents: List[IntentData],
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Process multiple intent classification results in batch.
    """
    try:
        user_id = current_user["uid"]
        results = []
        
        for intent_data in intents:
            try:
                result = await intent_processor.process_intent(
                    intent_data=intent_data.dict(),
                    user_id=user_id
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing intent in batch: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'intent': intent_data.intent
                })
        
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        return {
            'total_processed': len(results),
            'successful': successful,
            'failed': failed,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in process_batch_intents endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process batch intents: {str(e)}"
        )

@router.get("/supported-intents")
async def get_supported_intents():
    """
    Get list of supported intents and their expected entities.
    """
    return {
        "supported_intents": {
            "create_reminder": {
                "description": "Create a new reminder",
                "entities": {
                    "time": "Time for the reminder (optional)",
                    "person": "Person related to the reminder (optional)",
                    "date": "Date for the reminder (optional)"
                },
                "examples": [
                    "Remind me to call John at 3 PM",
                    "Set a reminder for the meeting tomorrow",
                    "Don't forget to buy groceries"
                ]
            },
            "create_note": {
                "description": "Create a new note",
                "entities": {},
                "examples": [
                    "Note: Meeting notes from today",
                    "Write down this important information",
                    "Save this for later"
                ]
            },
            "create_ledger": {
                "description": "Create a ledger entry for money owed/owing",
                "entities": {
                    "amount": "Monetary amount",
                    "person": "Person involved in the transaction"
                },
                "examples": [
                    "John owes me $50",
                    "I owe Sarah 25 dollars",
                    "Lent Mike $100"
                ]
            },
            "add_expense": {
                "description": "Add an expense (same as create_ledger)",
                "entities": {
                    "amount": "Monetary amount",
                    "person": "Person involved in the expense"
                },
                "examples": [
                    "Added expense: John owes $30",
                    "Track that Tom borrowed $75"
                ]
            },
            "chit_chat": {
                "description": "General conversation/chat",
                "entities": {},
                "examples": [
                    "Hello, how are you?",
                    "Thanks for your help",
                    "What can you do?"
                ]
            },
            "general_query": {
                "description": "General questions (treated as chat)",
                "entities": {},
                "examples": [
                    "What time is it?",
                    "How's the weather?",
                    "Help me with something"
                ]
            }
        }
    }

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