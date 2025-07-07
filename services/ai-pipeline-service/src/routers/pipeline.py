from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import httpx
from ..config import settings

router = APIRouter()

class TextRequest(BaseModel):
    text: str
    customer_id: str | None = None
    multi_intent: bool = True
    voice: str | None = None  # optional TTS voice

class PipelineResponse(BaseModel):
    reply_text: str
    intent: str | None = None
    audio_url: str | None = None

async def call_stt(audio: UploadFile):
    async with httpx.AsyncClient(base_url=settings.STT_URL, timeout=settings.TIMEOUT) as client:
        resp = await client.post("/stt/transcribe", files={"audio": (audio.filename, await audio.read(), audio.content_type)})
    resp.raise_for_status()
    return resp.json()["text"]

async def call_intent(text: str, multi: bool):
    async with httpx.AsyncClient(base_url=settings.INTENT_URL, timeout=settings.TIMEOUT) as client:
        resp = await client.post("/intent/classify", json={"text": text, "multi_intent": multi})
    resp.raise_for_status()
    data = resp.json()
    return data["primary_intent"], data.get("intents", [])

async def call_chat(text: str, customer_id: str | None):
    async with httpx.AsyncClient(base_url=settings.CHAT_URL, timeout=settings.TIMEOUT) as client:
        resp = await client.post("/chat/reply", json={"text": text, "customer_id": customer_id})
    resp.raise_for_status()
    return resp.json()["reply"]

async def call_tts(text: str, voice: str | None):
    async with httpx.AsyncClient(base_url=settings.TTS_URL, timeout=settings.TIMEOUT) as client:
        resp = await client.post("/tts/synthesize", json={"text": text, "voice": voice or "default"})
    resp.raise_for_status()
    # assuming service returns a presigned URL to the audio file
    return resp.json()["audio_url"]

@router.post("/transcribe-and-respond", response_model=PipelineResponse)
async def transcribe_and_respond(audio: UploadFile = File(...), customer_id: str | None = None, voice: str | None = None):
    """Full pipeline: audio -> text -> intent -> reply -> (optional) TTS"""
    try:
        text = await call_stt(audio)
        intent, _ = await call_intent(text, True)

        if intent == "chit_chat":
            reply = await call_chat(text, customer_id)
        else:
            reply = f"Intent '{intent}' acknowledged (CRUD call omitted in demo)."

        audio_url = await call_tts(reply, voice) if voice else None
        return PipelineResponse(reply_text=reply, intent=intent, audio_url=audio_url)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Downstream AI service error: {e}")

@router.post("/process", response_model=PipelineResponse)
async def process_text(req: TextRequest):
    """Text-only pipeline (no STT)"""
    intent, _ = await call_intent(req.text, req.multi_intent)
    if intent == "chit_chat":
        reply = await call_chat(req.text, req.customer_id)
    else:
        reply = f"Intent '{intent}' acknowledged (CRUD call omitted in demo)."
    audio_url = await call_tts(reply, req.voice) if req.voice else None
    return PipelineResponse(reply_text=reply, intent=intent, audio_url=audio_url) 
 
 