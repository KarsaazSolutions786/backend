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
    DATABASE_URL: str = "postgresql://postgres:admin123@localhost:5432/eindr"

    
    # AI Models paths (assuming local models)
    COQUI_STT_MODEL_PATH: str = "./models/coqui-stt.tflite"
    TTS_MODEL_PATH: str = "./models/tts"
    INTENT_MODEL_PATH: str = "./models/intent"
    CHAT_MODEL_PATH: str = "./models/bloom-560m"
    
    # Audio settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_BIT_DEPTH: int = 16
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"
    
    # Supported audio formats for STT
    SUPPORTED_AUDIO_FORMATS: List[str] = [".wav", ".wave"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True) 