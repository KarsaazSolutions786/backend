from pydantic import BaseModel
from typing import List
import os

class Settings(BaseModel):
    """Application settings and configuration."""
    
    # App settings
    APP_NAME: str = "Eindr Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./eindr.db"
    
    # AI Models paths (assuming local models)
    VOSK_MODEL_PATH: str = "./models/vosk-model-en-us-0.22"
    TTS_MODEL_PATH: str = "./models/tts"
    INTENT_MODEL_PATH: str = "./models/intent"
    CHAT_MODEL_PATH: str = "./models/bloom-560m"
    
    # Audio settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True) 