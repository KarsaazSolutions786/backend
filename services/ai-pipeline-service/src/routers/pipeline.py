from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import logging
import base64
from typing import Optional, Dict, Any
from ..config import settings
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from fastapi.responses import JSONResponse
import io
# Use secure shared authentication 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../shared'))

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

# Initialize auth service
# auth_service = AuthService() # This line is removed as per the new_code, as the import is removed.

class TextRequest(BaseModel):
    text: str
    customer_id: str = "default_user"
    language: str = "en"
    voice: Optional[str] = "default"
    speed: float = 1.0

class PipelineResponse(BaseModel):
    reply_text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    audio_available: bool = False
    audio_size: Optional[int] = None
    audio_content_type: Optional[str] = None
    suggested_action: Optional[str] = None
    data_saved: bool = False
    saved_data_info: Optional[Dict[str, Any]] = None

class AudioPipelineResponse(BaseModel):
    reply_text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    suggested_action: Optional[str] = None
    data_saved: bool = False
    saved_data_info: Optional[Dict[str, Any]] = None

async def call_stt(audio: UploadFile, customer_id: str = "default_user", language: str = "auto"):
    """Call STT service to transcribe audio"""
    try:
        async with httpx.AsyncClient(base_url=settings.STT_URL, timeout=settings.TIMEOUT) as client:
            files = {"audio": (audio.filename, await audio.read(), audio.content_type)}
            data = {"customer_id": customer_id, "language": language, "include_segments": "false"}
            resp = await client.post("/stt/transcribe", files=files, data=data)
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"STT transcription: {result.get('text', '')[:50]}...")
            return result.get("text", "")
    except httpx.HTTPError as e:
        logger.error(f"STT service error: {e}")
        raise HTTPException(status_code=502, detail=f"STT service error: {e}")

async def call_intent(text: str, customer_id: str = "default_user"):
    """Call Intent service to classify text"""
    try:
        async with httpx.AsyncClient(base_url=settings.INTENT_URL, timeout=settings.TIMEOUT) as client:
            resp = await client.post("/intent/classify", json={"text": text, "customer_id": customer_id})
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Intent classification: {result.get('intent', '')} (confidence: {result.get('confidence', 0):.2f})")
            return result
    except httpx.HTTPError as e:
        logger.error(f"Intent service error: {e}")
        raise HTTPException(status_code=502, detail=f"Intent service error: {e}")

async def call_chat(text: str, customer_id: str = "default_user"):
    """Call Chat service for conversation"""
    try:
        async with httpx.AsyncClient(base_url=settings.CHAT_URL, timeout=settings.TIMEOUT) as client:
            resp = await client.post("/conversations/chat", json={"message": text, "customer_id": customer_id})
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Chat response: {result.get('response', '')[:50]}...")
            return result.get("response", "")
    except httpx.HTTPError as e:
        logger.error(f"Chat service error: {e}")
        # Fallback response if chat service is unavailable
        return f"I understand you said: '{text}'. How can I help you with that?"

async def call_tts(text: str, language: str = "en", voice: str = "default", speed: float = 1.0):
    """Call TTS service to synthesize speech"""
    try:
        async with httpx.AsyncClient(base_url=settings.TTS_URL, timeout=settings.TIMEOUT) as client:
            resp = await client.post("/tts/synthesize", json={
                "text": text, 
                "language": language, 
                "voice": voice, 
                "speed": speed
            })
            resp.raise_for_status()
            # TTS service returns audio data directly
            audio_data = resp.content
            content_type = resp.headers.get("content-type", "audio/wav")
            logger.info(f"TTS synthesis completed: {len(audio_data)} bytes")
            return audio_data, content_type
    except httpx.HTTPError as e:
        logger.error(f"TTS service error: {e}")
        raise HTTPException(status_code=502, detail=f"TTS service error: {e}")

async def save_data_based_on_intent(intent: str, text: str, customer_id: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """Save data to appropriate database tables based on intent"""
    try:
        data_saved = False
        saved_data_info = {}
        
        # Get authentication token
        # token = await auth_service.get_valid_token(customer_id) # This line is removed as per the new_code, as the import is removed.
        # if not token: # This line is removed as per the new_code, as the import is removed.
        #     logger.warning("Could not get authentication token - data not saved") # This line is removed as per the new_code, as the import is removed.
        #     return { # This line is removed as per the new_code, as the import is removed.
        #         "data_saved": False, # This line is removed as per the new_code, as the import is removed.
        #         "saved_data_info": {"error": "Authentication token not available"} # This line is removed as per the new_code, as the import is removed.
        #     } # This line is removed as per the new_code, as the import is removed.
        
        # headers = auth_service.get_auth_headers(token) # This line is removed as per the new_code, as the import is removed.
        
        if intent == "create_reminder":
            # Save to reminder service
            reminder_data = {
                "title": entities.get("title", "Reminder"),
                "description": text,
                "time": entities.get("time", "2024-01-01T12:00:00Z"),  # Default time if not specified
                "repeat_pattern": "none",
                "timezone": "UTC",
                "priority": entities.get("priority", "medium"),
                "category": entities.get("category", "general")
            }
            
            async with httpx.AsyncClient(timeout=settings.TIMEOUT) as client:
                resp = await client.post("http://reminder-service:8000/reminders/", json=reminder_data) # This line is removed as per the new_code, as the import is removed.
                if resp.status_code == 200 or resp.status_code == 201:
                    data_saved = True
                    saved_data_info = resp.json()
                    logger.info(f"Reminder saved: {saved_data_info}")
                elif resp.status_code == 403:
                    logger.warning("Reminder service requires authentication - data not saved")
                    saved_data_info = {"error": "Authentication required", "status": 403}
                else:
                    logger.error(f"Failed to save reminder: {resp.status_code} - {resp.text}")
                    saved_data_info = {"error": f"Service error: {resp.status_code}", "status": resp.status_code}
        
        elif intent == "create_note":
            # Save to note service
            note_data = {
                "title": entities.get("title", "Note"),
                "description": text,
                "content_type": "text"
            }
            
            async with httpx.AsyncClient(timeout=settings.TIMEOUT) as client:
                resp = await client.post("http://note-service:8000/notes/", json=note_data) # This line is removed as per the new_code, as the import is removed.
                if resp.status_code == 200 or resp.status_code == 201:
                    data_saved = True
                    saved_data_info = resp.json()
                    logger.info(f"Note saved: {saved_data_info}")
                elif resp.status_code == 403:
                    logger.warning("Note service requires authentication - data not saved")
                    saved_data_info = {"error": "Authentication required", "status": 403}
                else:
                    logger.error(f"Failed to save note: {resp.status_code} - {resp.text}")
                    saved_data_info = {"error": f"Service error: {resp.status_code}", "status": resp.status_code}
        
        elif intent == "add_expense":
            # Save to ledger service
            amount = entities.get("amount", 0)
            if amount > 0:
                # For self-expenses, use the same customer_id for both customer and friend
                expense_data = {
                    "friend_id": int(customer_id),  # Same as customer_id for self-expenses
                    "amount": float(amount),
                    "ledger_direction_id": 2,  # outgoing (assuming 2 is outgoing)
                    "notes": text
                }
                
                async with httpx.AsyncClient(timeout=settings.TIMEOUT) as client:
                    resp = await client.post("http://ledger-service:8000/ledger-entries/", json=expense_data) # This line is removed as per the new_code, as the import is removed.
                    if resp.status_code == 200 or resp.status_code == 201:
                        data_saved = True
                        saved_data_info = resp.json()
                        logger.info(f"Expense saved: {saved_data_info}")
                    elif resp.status_code == 403:
                        logger.warning("Ledger service requires authentication - data not saved")
                        saved_data_info = {"error": "Authentication required", "status": 403}
                    elif resp.status_code == 400:
                        logger.error(f"Ledger service validation error: {resp.text}")
                        saved_data_info = {"error": f"Validation error: {resp.text}", "status": 400}
                    else:
                        logger.error(f"Failed to save expense: {resp.status_code} - {resp.text}")
                        saved_data_info = {"error": f"Service error: {resp.status_code}", "status": resp.status_code}
            else:
                logger.warning("No valid amount found for expense - not saved")
                saved_data_info = {"error": "No valid amount specified"}
        
        return {
            "data_saved": data_saved,
            "saved_data_info": saved_data_info
        }
        
    except Exception as e:
        logger.error(f"Error saving data based on intent: {e}")
        return {
            "data_saved": False,
            "saved_data_info": {"error": str(e)}
        }

@router.post("/audio-pipeline", response_model=AudioPipelineResponse)
async def audio_pipeline(
    audio: UploadFile = File(...),
    request: Request = None,
    current_customer_id: int = Depends(get_current_customer_id),
    language: str = Form("auto"),
    voice: str = Form("default"),
    speed: float = Form(1.0)
):
    """Complete audio processing pipeline: STT -> Intent -> Chat -> Save Data"""
    # Get limiter from app state to avoid circular import
    limiter = request.app.state.limiter
    
    # Apply rate limiting
    await limiter.check_request_and_update(request)
    
    try:
        # Step 1: Speech-to-Text
        logger.info("Starting audio pipeline...")
        transcribed_text = await call_stt(audio, str(current_customer_id), language)
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Step 2: Intent Classification
        intent_result = await call_intent(transcribed_text, str(current_customer_id))
        intent = intent_result.get("intent", "unknown")
        confidence = intent_result.get("confidence", 0.0)
        entities = intent_result.get("entities", {})
        
        # Step 3: Chat Response
        chat_response = await call_chat(transcribed_text, str(current_customer_id))
        
        # Step 4: Save data based on intent
        save_result = await save_data_based_on_intent(intent, transcribed_text, str(current_customer_id), entities)
        
        # Determine suggested action
        suggested_action = None
        if intent == "create_reminder":
            suggested_action = "reminder_created"
        elif intent == "create_note":
            suggested_action = "note_created"
        elif intent == "add_expense":
            suggested_action = "expense_recorded"
        elif intent == "greeting":
            suggested_action = "greeting_responded"
        
        logger.info(f"Audio pipeline completed successfully for user {current_customer_id}")
        
        return AudioPipelineResponse(
            reply_text=chat_response,
            intent=intent,
            confidence=confidence,
            entities=entities,
            suggested_action=suggested_action,
            data_saved=save_result["data_saved"],
            saved_data_info=save_result["saved_data_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

@router.post("/text-pipeline", response_model=PipelineResponse)
async def text_pipeline(
    req: TextRequest, 
    request: Request = None,
    current_customer_id: int = Depends(get_current_customer_id)
):
    """Complete text processing pipeline: Intent -> Chat -> TTS -> Save Data"""
    # Get limiter from app state to avoid circular import
    limiter = request.app.state.limiter
    
    # Apply rate limiting
    await limiter.check_request_and_update(request)
    
    try:
        logger.info("Starting text pipeline...")
        
        # Step 1: Intent Classification
        intent_result = await call_intent(req.text, req.customer_id)
        intent = intent_result.get("intent", "unknown")
        confidence = intent_result.get("confidence", 0.0)
        entities = intent_result.get("entities", {})
        
        # Step 2: Chat Response
        chat_response = await call_chat(req.text, req.customer_id)
        
        # Step 3: Text-to-Speech (optional)
        audio_data = None
        audio_size = None
        audio_content_type = None
        audio_available = False
        
        try:
            audio_data, audio_content_type = await call_tts(chat_response, req.language, req.voice, req.speed)
            audio_size = len(audio_data) if audio_data else 0
            audio_available = bool(audio_data)
        except Exception as e:
            logger.warning(f"TTS failed, continuing without audio: {e}")
        
        # Step 4: Save data based on intent
        save_result = await save_data_based_on_intent(intent, req.text, req.customer_id, entities)
        
        # Determine suggested action
        suggested_action = None
        if intent == "create_reminder":
            suggested_action = "reminder_created"
        elif intent == "create_note":
            suggested_action = "note_created"
        elif intent == "add_expense":
            suggested_action = "expense_recorded"
        elif intent == "greeting":
            suggested_action = "greeting_responded"
        
        logger.info(f"Text pipeline completed successfully")
        
        return PipelineResponse(
            reply_text=chat_response,
            intent=intent,
            confidence=confidence,
            entities=entities,
            audio_available=audio_available,
            audio_size=audio_size,
            audio_content_type=audio_content_type,
            suggested_action=suggested_action,
            data_saved=save_result["data_saved"],
            saved_data_info=save_result["saved_data_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

@router.post("/synthesize-audio")
async def synthesize_audio(req: TextRequest):
    """Direct text-to-speech synthesis"""
    try:
        audio_data, content_type = await call_tts(req.text, req.language, req.voice, req.speed)
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=content_type,
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@router.get("/services/status")
async def get_services_status():
    """Check status of all AI services"""
    services = {
        "stt": {"url": settings.STT_URL, "status": "unknown"},
        "intent": {"url": settings.INTENT_URL, "status": "unknown"}, 
        "chat": {"url": settings.CHAT_URL, "status": "unknown"},
        "tts": {"url": settings.TTS_URL, "status": "unknown"}
    }
    
    # Check each service
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, service_info in services.items():
            try:
                resp = await client.get(f"{service_info['url']}/health")
                if resp.status_code == 200:
                    service_info["status"] = "healthy"
                else:
                    service_info["status"] = f"error_{resp.status_code}"
            except:
                service_info["status"] = "unreachable"
    
    all_healthy = all(s["status"] == "healthy" for s in services.values())
    
    return {
        "pipeline_status": "healthy" if all_healthy else "degraded",
        "services": services,
        "timestamp": "2024-01-01T00:00:00Z"  # You can use real timestamp here
    } 
 
 