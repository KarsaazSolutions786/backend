from pydantic import BaseModel
from typing import List
import os

class Settings(BaseModel):
    """Application settings and configuration."""
    
    # App settings
    APP_NAME: str = "Eindr Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")  # Railway needs 0.0.0.0
    PORT: int = int(os.getenv("PORT", "8000"))  # Railway sets PORT env var
    
    # Environment detection
    IS_RAILWAY: bool = os.getenv("RAILWAY_ENVIRONMENT") is not None
    MINIMAL_MODE: bool = os.getenv("MINIMAL_MODE", "false").lower() == "true"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS - More permissive for Railway
    ALLOWED_HOSTS: List[str] = ["*"] if os.getenv("RAILWAY_ENVIRONMENT") else ["localhost", "127.0.0.1"]
    
    # Database - Railway provides DATABASE_URL automatically for PostgreSQL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:admin123@localhost:5432/eindr")

    # Development mode flag
    DEV_MODE: bool = os.getenv("DEV_MODE", "false").lower() == "true"
    
    # AI Models paths (only used when not in minimal mode)
    COQUI_STT_MODEL_PATH: str = "./models/coqui-stt.tflite"
    TTS_MODEL_PATH: str = "./models/tts"
    INTENT_MODEL_PATH: str = "./models/intent"
    CHAT_MODEL_PATH: str = "./models/bloom-560m"
    
    # Audio settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_BIT_DEPTH: int = 16
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
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