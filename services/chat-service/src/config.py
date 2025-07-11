from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # App settings
    SERVICE_NAME: str = "chat-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://eindr:eindr_pass@new-postgres-server:5432/eindr_db")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
    
    # RabbitMQ
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    
    # Auth service
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8000")
    
    # BLOOM Model settings
    BLOOM_MODEL_PATH: str = os.getenv("BLOOM_MODEL_PATH", "/app/models/bloom-560m")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/models")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
