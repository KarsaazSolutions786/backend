from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://eindr:eindr_pass@new-postgres-server:5432/eindr_db")
    
    # Service settings
    service_name: str = os.getenv("SERVICE_NAME", "tts-service")
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "0.0.0.0")
    
    # External services
    auth_service_url: str = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Model settings
    model_path: str = os.getenv("MODEL_PATH", "/app/models")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"

settings = Settings()

settings = Settings()
