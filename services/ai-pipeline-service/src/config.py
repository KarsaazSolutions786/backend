import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    STT_URL: str = os.getenv("STT_SERVICE_URL", "http://localhost:8008")
    INTENT_URL: str = os.getenv("INTENT_SERVICE_URL", "http://localhost:8010")
    CHAT_URL: str = os.getenv("CHAT_SERVICE_URL", "http://localhost:8011")
    TTS_URL: str = os.getenv("TTS_SERVICE_URL", "http://localhost:8009")
    AUTH_URL: str = os.getenv("AUTH_SERVICE_URL", "http://localhost:8002")
    TIMEOUT: int = int(os.getenv("AI_HTTP_TIMEOUT", "120"))
    
    # JWT settings for service-to-service authentication
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Service account credentials for internal communication
    SERVICE_ACCOUNT_EMAIL: str = os.getenv("SERVICE_ACCOUNT_EMAIL", "ai-pipeline@eindr.com")
    SERVICE_ACCOUNT_PASSWORD: str = os.getenv("SERVICE_ACCOUNT_PASSWORD", "service-password")

settings = Settings() 
 
 