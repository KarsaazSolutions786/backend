from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

# Try to load from local .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    # App settings
    SERVICE_NAME: str = "auth-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://eindr:eindr_pass@new-postgres-server:5432/eindr_db")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
    
    # JWT Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Legacy JWT settings for compatibility
    JWT_SECRET: str = os.getenv("JWT_SECRET", SECRET_KEY)
    JWT_ALGORITHM: str = ALGORITHM
    
    # Security
    PASSWORD_MIN_LENGTH: int = 8
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    
    @property
    def allowed_origins_list(self) -> List[str]:
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings():
    return Settings()

settings = Settings()