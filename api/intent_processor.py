"""
FastAPI router for PyTorch-based intent processing
Handles classification, training, and database integration.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import aiofiles
import asyncio
from pathlib import Path
import os

# Import dependencies
from firebase_auth import verify_firebase_token
from services.database_integration_service import DatabaseIntegrationService
from utils.logger import logger

# Initialize router
router = APIRouter()

# Initialize services conditionally
db_integration_service = DatabaseIntegrationService()

# Global service variables - will be initialized on first use
pytorch_intent_service = None
intent_service = None

def get_intent_service():
    """Get the appropriate intent service based on mode and availability."""
    global pytorch_intent_service, intent_service
    
    # Check if we're in minimal mode or Railway environment
    is_minimal_mode = os.getenv("MINIMAL_MODE", "false").lower() == "true"
    is_railway_env = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    # Force minimal mode in Railway
    if is_railway_env:
        is_minimal_mode = True
    
    if is_minimal_mode:
        # Use lightweight intent service in minimal mode or Railway
        if intent_service is None:
            logger.info("Initializing lightweight intent service for API (minimal/Railway mode)")
            from services.intent_service import IntentService
            intent_service = IntentService()
        return intent_service
    else:
        # Try to use PyTorch service in full mode (local only)
        if pytorch_intent_service is None:
            try:
                logger.info("Initializing PyTorch intent service for API")
                from services.pytorch_intent_service import PyTorchIntentService
                pytorch_intent_service = PyTorchIntentService()
            except Exception as e:
                logger.warning(f"Failed to initialize PyTorch intent service: {e}")
                logger.info("Falling back to lightweight intent service for API")
                from services.intent_service import IntentService
                pytorch_intent_service = IntentService()  # Store in pytorch_intent_service var to avoid re-initialization
        return pytorch_intent_service

# Pydantic models for request/response
class IntentClassificationRequest(BaseModel):
    text: str
    multi_intent: bool = True
    include_entities: bool = True
    confidence_threshold: float = 0.1

class IntentClassificationResponse(BaseModel):
    success: bool
    intent: Optional[str] = None
    intents: Optional[List[str]] = None
    confidence: Optional[float] = None
    overall_confidence: Optional[float] = None
    entities: Optional[Dict[str, Any]] = None
    original_text: str
    model_used: str
    type: str  # "single_intent" or "multi_intent"
    segments: Optional[List[str]] = None
    results: Optional[List[Dict[str, Any]]] = None

class TrainingRequest(BaseModel):
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    data_source: str = "generated"  # "generated" or "file"
    data_file_path: Optional[str] = None

class DatabaseStorageRequest(BaseModel):
    classification_result: Dict[str, Any]
    store_in_database: bool = True
    validate_user: bool = True

@router.post("/classify", response_model=IntentClassificationResponse)
async def classify_intent(
    request: IntentClassificationRequest,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Classify intent(s) in the provided text using available model.
    
    Features:
    - Single and multi-intent detection
    - Entity extraction
    - Confidence scoring
    - Fallback to rule-based classification
    """
    try:
        user_id = current_user["uid"]
        logger.info(f"Intent classification request from user {user_id}: '{request.text}'")
        
        # Get the appropriate intent service
        service = get_intent_service()
        
        # Perform classification
        result = await service.classify_intent(
            text=request.text,
            multi_intent=request.multi_intent
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Intent classification failed"
            )
        
        # Filter results by confidence threshold
        if request.confidence_threshold > 0:
            if result.get("type") == "multi_intent":
                # Filter individual results
                filtered_results = []
                for res in result.get("results", []):
                    if res.get("confidence", 0) >= request.confidence_threshold:
                        filtered_results.append(res)
                result["results"] = filtered_results
                result["intents"] = [r["intent"] for r in filtered_results]
            else:
                # Check single result confidence
                if result.get("confidence", 0) < request.confidence_threshold:
                    result["intent"] = "chit_chat"  # Fallback
                    result["confidence"] = 0.1
        
        # Format response
        response_data = {
            "success": True,
            "original_text": request.text,
            "model_used": result.get("model_used", "lightweight"),
            "type": result.get("type", "single_intent")
        }
        
        if result.get("type") == "multi_intent":
            response_data.update({
                "intents": result.get("intents", []),
                "overall_confidence": result.get("overall_confidence", 0.0),
                "segments": result.get("segments", []),
                "results": result.get("results", [])
            })
        else:
            response_data.update({
                "intent": result.get("intent"),
                "confidence": result.get("confidence", 0.0),
                "entities": result.get("entities", {}) if request.include_entities else None
            })
        
        logger.info(f"Classification successful: {response_data['type']}")
        return IntentClassificationResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

@router.post("/process-and-store")
async def process_and_store(
    request: DatabaseStorageRequest,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Process intent classification result and store in database.
    
    This endpoint:
    1. Takes classification results
    2. Validates user existence
    3. Stores data in appropriate tables
    4. Returns storage confirmation
    """
    try:
        user_id = current_user["uid"]
        logger.info(f"Processing and storing classification result for user {user_id}")
        
        # Validate user if requested
        if request.validate_user:
            await db_integration_service.create_user_if_not_exists(user_id)
        
        # Store classification result
        if request.store_in_database:
            storage_result = await db_integration_service.store_classification_result(
                classification_result=request.classification_result,
                user_id=user_id
            )
            
            if not storage_result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Database storage failed: {storage_result.get('error')}"
                )
            
            logger.info(f"Successfully stored classification result for user {user_id}")
            return {
                "success": True,
                "user_id": user_id,
                "storage_result": storage_result,
                "message": "Classification result stored successfully"
            }
        else:
            return {
                "success": True,
                "user_id": user_id,
                "message": "Classification processed but not stored (store_in_database=False)"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process and store error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Process and store failed: {str(e)}"
        )

@router.post("/classify-and-store")
async def classify_and_store(
    request: IntentClassificationRequest,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Complete pipeline: classify intent and store in database.
    
    This is the main endpoint that combines classification and storage.
    """
    try:
        user_id = current_user["uid"]
        logger.info(f"Full pipeline request from user {user_id}: '{request.text}'")
        
        # Step 1: Classify intent
        classification_result = await get_intent_service().classify_intent(
            text=request.text,
            multi_intent=request.multi_intent
        )
        
        if not classification_result:
            raise HTTPException(
                status_code=500,
                detail="Intent classification failed"
            )
        
        # Step 2: Ensure user exists
        await db_integration_service.create_user_if_not_exists(user_id)
        
        # Step 3: Store in database
        storage_result = await db_integration_service.store_classification_result(
            classification_result=classification_result,
            user_id=user_id
        )
        
        if not storage_result.get("success"):
            logger.warning(f"Database storage failed: {storage_result.get('error')}")
            # Don't fail the entire request if storage fails
        
        # Step 4: Format comprehensive response
        response = {
            "success": True,
            "user_id": user_id,
            "classification": classification_result,
            "database_storage": storage_result,
            "pipeline_steps": {
                "classification_completed": True,
                "user_validation_completed": True,
                "database_storage_completed": storage_result.get("success", False)
            }
        }
        
        logger.info(f"Full pipeline completed for user {user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classify and store pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

@router.post("/train-model")
async def train_model(
    request: TrainingRequest,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Train or fine-tune the intent classification model.
    
    Note: This is a resource-intensive operation and should be used carefully.
    """
    try:
        user_id = current_user["uid"]
        logger.info(f"Model training request from user {user_id}")
        
        # Check if user has permission to train model (you might want to add admin check)
        logger.info("Starting model training...")
        
        # Start training
        success = await get_intent_service().train_model(
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size
        )
        
        if success:
            logger.info("Model training completed successfully")
            return {
                "success": True,
                "message": "Model training completed successfully",
                "training_parameters": {
                    "epochs": request.epochs,
                    "learning_rate": request.learning_rate,
                    "batch_size": request.batch_size
                },
                "model_info": get_intent_service().get_model_info()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Model training failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@router.get("/model-info")
async def get_model_info(
    current_user: dict = Depends(verify_firebase_token)
):
    """Get information about the current model and capabilities."""
    try:
        model_info = get_intent_service().get_model_info()
        return {
            "success": True,
            "model_info": model_info,
            "supported_intents": get_intent_service().intent_labels,
            "capabilities": {
                "single_intent_classification": True,
                "multi_intent_classification": True,
                "entity_extraction": True,
                "confidence_scoring": True,
                "database_integration": True,
                "model_training": model_info.get("pytorch_available", False)
            }
        }
        
    except Exception as e:
        logger.error(f"Get model info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.get("/user-statistics")
async def get_user_statistics(
    current_user: dict = Depends(verify_firebase_token)
):
    """Get user's usage statistics across all intent categories."""
    try:
        user_id = current_user["uid"]
        stats = await db_integration_service.get_user_statistics(user_id)
        
        if not stats.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get statistics: {stats.get('error')}"
            )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user statistics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user statistics: {str(e)}"
        )

@router.get("/recent-records/{table_name}")
async def get_recent_records(
    table_name: str,
    limit: int = 10,
    current_user: dict = Depends(verify_firebase_token)
):
    """Get recent records for a user from a specific table."""
    try:
        user_id = current_user["uid"]
        
        # Validate table name
        supported_tables = db_integration_service.get_supported_tables()
        if table_name not in supported_tables:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported table. Supported tables: {supported_tables}"
            )
        
        records = await db_integration_service.get_recent_records(
            user_id=user_id,
            table_name=table_name,
            limit=min(limit, 50)  # Cap at 50 records
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "table_name": table_name,
            "records": records,
            "count": len(records)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get recent records error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent records: {str(e)}"
        )

@router.post("/batch-classify")
async def batch_classify(
    texts: List[str],
    multi_intent: bool = True,
    store_results: bool = False,
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Classify multiple texts in batch for efficiency.
    """
    try:
        user_id = current_user["uid"]
        logger.info(f"Batch classification request from user {user_id}: {len(texts)} texts")
        
        if len(texts) > 100:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 100 texts per request."
            )
        
        results = []
        
        # Process each text
        for i, text in enumerate(texts):
            try:
                result = await get_intent_service().classify_intent(
                    text=text,
                    multi_intent=multi_intent
                )
                
                if store_results and result:
                    # Store in database
                    storage_result = await db_integration_service.store_classification_result(
                        classification_result=result,
                        user_id=user_id
                    )
                    result["database_stored"] = storage_result.get("success", False)
                
                results.append({
                    "index": i,
                    "text": text,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append({
                    "index": i,
                    "text": text,
                    "result": None,
                    "success": False,
                    "error": str(e)
                })
        
        successful_count = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "user_id": user_id,
            "total_texts": len(texts),
            "successful_classifications": successful_count,
            "failed_classifications": len(texts) - successful_count,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch classification failed: {str(e)}"
        )

@router.post("/upload-training-data")
async def upload_training_data(
    file: UploadFile = File(...),
    current_user: dict = Depends(verify_firebase_token)
):
    """
    Upload training data file for model training.
    Supports CSV and JSON formats.
    """
    try:
        user_id = current_user["uid"]
        logger.info(f"Training data upload from user {user_id}: {file.filename}")
        
        # Validate file format
        if not file.filename.endswith(('.csv', '.json')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use CSV or JSON."
            )
        
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{user_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"Training data saved to {file_path}")
        
        return {
            "success": True,
            "user_id": user_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "message": "Training data uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload training data error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for the intent processing service."""
    try:
        model_ready = get_intent_service().is_ready()
        model_info = get_intent_service().get_model_info()
        
        return {
            "status": "healthy" if model_ready else "degraded",
            "model_ready": model_ready,
            "pytorch_available": model_info.get("pytorch_available", False),
            "transformers_available": model_info.get("transformers_available", False),
            "device": model_info.get("device", "unknown"),
            "supported_intents": get_intent_service().intent_labels
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Export the router
__all__ = ["router"] 