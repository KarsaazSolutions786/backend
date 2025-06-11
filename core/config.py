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
    ALLOWED_HOSTS: List[str] = ["*"]  # Allow all hosts during development
    
    # Database - Railway provides DATABASE_URL automatically for PostgreSQL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:admin123@localhost:5432/eindr")

    # Development mode flag
    DEV_MODE: bool = os.getenv("DEV_MODE", "false").lower() == "true"
    
    # AI Models paths (only used when not in minimal mode)
    COQUI_STT_MODEL_PATH: str = "./models/coqui-stt.tflite"
    TTS_MODEL_PATH: str = "./models/tts"
    COQUI_TTS_MODEL_PATH: str = "./models/coqui.tflite"
    INTENT_MODEL_PATH: str = "./models/intent"
    CHAT_MODEL_PATH: str = "./models/Bloom560m.bin"
    
    # Chat Service Configuration (Bloom-560M via vLLM)
    CHAT_MODEL_NAME: str = os.getenv("CHAT_MODEL_NAME", "bigscience/bloom-560m")
    VLLM_SERVER_URL: str = os.getenv("VLLM_SERVER_URL", "http://localhost:8001")
    VLLM_SERVER_PORT: int = int(os.getenv("VLLM_SERVER_PORT", "8001"))
    
    # Chat generation parameters
    CHAT_MAX_TOKENS: int = int(os.getenv("CHAT_MAX_TOKENS", "150"))
    CHAT_TEMPERATURE: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
    CHAT_TOP_P: float = float(os.getenv("CHAT_TOP_P", "0.9"))
    CHAT_TOP_K: int = int(os.getenv("CHAT_TOP_K", "50"))
    CHAT_REPETITION_PENALTY: float = float(os.getenv("CHAT_REPETITION_PENALTY", "1.1"))
    CHAT_STOP_SEQUENCES: List[str] = ["User:", "\nUser:", "System:", "\nSystem:"]
    
    # vLLM Configuration
    VLLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8"))
    VLLM_MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "2048"))
    VLLM_TENSOR_PARALLEL_SIZE: int = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    VLLM_ENFORCE_EAGER: bool = os.getenv("VLLM_ENFORCE_EAGER", "false").lower() == "true"
    
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