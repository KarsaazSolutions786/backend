"""
Standardized CORS Configuration for Eindr Microservices
"""
import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

class CORSConfig:
    """Centralized CORS configuration"""
    
    # Environment-based CORS origins
    DEVELOPMENT_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:8080"
    ]
    
    PRODUCTION_ORIGINS = [
        "https://yourdomain.com",
        "https://app.yourdomain.com",
        "https://api.yourdomain.com"
    ]
    
    @classmethod
    def get_allowed_origins(cls) -> List[str]:
        """Get allowed origins based on environment"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            # In production, use secure origins from environment variable
            origins_str = os.getenv("ALLOWED_ORIGINS", "")
            if origins_str:
                return [origin.strip() for origin in origins_str.split(",")]
            return cls.PRODUCTION_ORIGINS
        else:
            # In development, use permissive localhost origins
            return cls.DEVELOPMENT_ORIGINS

def setup_cors(app: FastAPI, service_name: str = "eindr-service"):
    """
    Set up CORS middleware for a FastAPI app
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service for logging
    """
    allowed_origins = CORSConfig.get_allowed_origins()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept",
            "Origin",
            "User-Agent",
            "DNT",
            "Cache-Control",
            "X-Mx-ReqToken",
            "Keep-Alive",
            "If-Modified-Since"
        ],
        expose_headers=["X-Process-Time"],
        max_age=600  # 10 minutes
    )
    
    # Log CORS configuration (in development only)
    if os.getenv("ENVIRONMENT", "development").lower() == "development":
        print(f"[{service_name}] CORS configured with origins: {allowed_origins}")

# Legacy function for backward compatibility
def get_cors_origins() -> List[str]:
    """Get CORS origins - legacy function"""
    return CORSConfig.get_allowed_origins() 