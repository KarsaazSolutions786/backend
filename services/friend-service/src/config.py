from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # App settings
    SERVICE_NAME: str = "friend-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./friend_service.db")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
    
    # RabbitMQ
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    
    # Auth service
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8000")
    
    # JWT Settings - Must match auth service configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "eindr-super-secure-jwt-secret-key-for-production-2024-v1")
    ALGORITHM: str = "HS256"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
