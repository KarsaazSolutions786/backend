"""
Shared Security Configuration for Eindr Microservices

This module provides centralized security configurations that all services can use
for consistent security settings across the platform.
"""

import os
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

def _get_allowed_origins() -> List[str]:
    """Get allowed CORS origins based on environment"""
    environment = os.getenv("ENVIRONMENT", "development")
    is_production = environment == "production"
    is_development = environment == "development"
    env_origins = os.getenv("ALLOWED_ORIGINS", "")
    
    if is_production:
        if not env_origins or env_origins == "*":
            raise ValueError("Production environment requires explicit ALLOWED_ORIGINS")
        origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        for origin in origins:
            if "*" in origin:
                raise ValueError(f"Wildcard origin not allowed in production: {origin}")
        return origins
    elif is_development:
        if env_origins:
            return [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        return [
            "http://localhost:3000", "http://localhost:3001", "http://localhost:8080",
            "http://localhost:8081", "http://localhost:4200", "http://127.0.0.1:3000"
        ]
    else:
        if not env_origins:
            raise ValueError(f"Environment {environment} requires explicit ALLOWED_ORIGINS")
        return [origin.strip() for origin in env_origins.split(",") if origin.strip()]

class SecurityConfig:
    """Centralized security configuration"""
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    IS_PRODUCTION = ENVIRONMENT == "production"
    IS_DEVELOPMENT = ENVIRONMENT == "development"
    ALLOWED_ORIGINS = _get_allowed_origins()
    ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    ALLOWED_HEADERS = ["Authorization", "Content-Type", "X-Requested-With", "X-CSRF-Token", "X-Request-ID"]
    RATE_LIMIT_ENABLED = True
    DEFAULT_RATE_LIMIT = "100/minute"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if IS_PRODUCTION else "DEBUG")
    LOG_SENSITIVE_DATA = False
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY", 
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains" if IS_PRODUCTION else None,
        "Content-Security-Policy": "default-src 'self'" if IS_PRODUCTION else None
    }

def setup_cors(app, service_name: str = "microservice"):
    """Setup CORS middleware for a FastAPI application"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SecurityConfig.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=SecurityConfig.ALLOWED_METHODS,
        allow_headers=SecurityConfig.ALLOWED_HEADERS,
        expose_headers=["X-Request-ID", "X-Process-Time"]
    )
    logger.info(f"CORS middleware configured for {service_name}")

def add_security_headers_middleware(app):
    """Add security headers middleware to FastAPI app"""
    @app.middleware("http")
    async def security_headers_middleware(request, call_next):
        response = await call_next(request)
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            if value:
                response.headers[header] = value
        return response

class SensitiveDataFilter:
    """Filter to prevent logging of sensitive data"""
    SENSITIVE_FIELDS = {
        'password', 'password_hash', 'token', 'secret', 'key', 
        'authorization', 'auth', 'jwt', 'session', 'cookie',
        'api_key', 'access_token', 'refresh_token', 'bearer'
    }
    
    @classmethod
    def filter_sensitive_data(cls, data: Any) -> Any:
        """Filter sensitive data from logs"""
        if isinstance(data, dict):
            filtered = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in cls.SENSITIVE_FIELDS):
                    filtered[key] = "[REDACTED]"
                else:
                    filtered[key] = cls.filter_sensitive_data(value)
            return filtered
        elif isinstance(data, list):
            return [cls.filter_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            if len(data) > 50 and data.replace(".", "").replace("_", "").replace("-", "").isalnum():
                return "[REDACTED_TOKEN]"
            return data
        else:
            return data

RATE_LIMIT_CONFIGS = {
    "auth": {"login": "5/minute", "register": "3/minute", "logout": "10/minute"},
    "api": {"read": "100/minute", "write": "50/minute", "delete": "20/minute"},
    "ai": {"stt": "30/minute", "tts": "30/minute", "chat": "20/minute"}
}

def get_rate_limit_for_endpoint(endpoint_type: str, action: str) -> str:
    """Get rate limit configuration for a specific endpoint"""
    return RATE_LIMIT_CONFIGS.get(endpoint_type, {}).get(action, SecurityConfig.DEFAULT_RATE_LIMIT) 