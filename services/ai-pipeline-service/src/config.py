import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    STT_URL: str = os.getenv("STT_SERVICE_URL", "http://localhost:8008")
    INTENT_URL: str = os.getenv("INTENT_SERVICE_URL", "http://localhost:8010")
    CHAT_URL: str = os.getenv("CHAT_SERVICE_URL", "http://localhost:8011")
    TTS_URL: str = os.getenv("TTS_SERVICE_URL", "http://localhost:8009")
    TIMEOUT: int = int(os.getenv("AI_HTTP_TIMEOUT", "30"))

settings = Settings() 
 
 