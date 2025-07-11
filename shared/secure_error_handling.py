"""
Secure Error Handling for Eindr Microservices

This module provides secure error handling that prevents information disclosure
while maintaining proper logging for debugging and monitoring.
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException as FastAPIHTTPException
from pydantic import ValidationError
import sqlalchemy.exc

logger = logging.getLogger(__name__)

class SecurityErrorHandler:
    """Centralized secure error handling"""
    
    # Generic error messages to prevent information disclosure
    GENERIC_MESSAGES = {
        400: "Invalid request. Please check your input and try again.",
        401: "Authentication required. Please log in and try again.",
        403: "Access denied. You don't have permission to access this resource.",
        404: "The requested resource was not found.",
        405: "Method not allowed for this endpoint.",
        409: "A conflict occurred. Please check your request and try again.",
        422: "Invalid input data. Please check your request format.",
        429: "Too many requests. Please try again later.",
        500: "An internal error occurred. Please try again later.",
        502: "Service temporarily unavailable. Please try again later.",
        503: "Service temporarily unavailable. Please try again later.",
        504: "Request timeout. Please try again later."
    }
    
    # Database error mappings
    DB_ERROR_MAPPINGS = {
        'IntegrityError': ('Constraint violation', 400),
        'DataError': ('Invalid data format', 400),
        'OperationalError': ('Database operation failed', 503),
        'ProgrammingError': ('Invalid database operation', 500),
        'NotSupportedError': ('Operation not supported', 501),
        'DatabaseError': ('Database error occurred', 503)
    }
    
    @classmethod
    def generate_error_id(cls) -> str:
        """Generate unique error ID for tracking"""
        return f"ERR-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    
    @classmethod
    def handle_database_error(cls, error: Exception, request: Request = None) -> JSONResponse:
        """
        Handle database errors securely
        
        Args:
            error: Database exception
            request: Optional request object
            
        Returns:
            JSONResponse with generic error message
        """
        error_id = cls.generate_error_id()
        error_type = type(error).__name__
        
        # Log the actual error with full details for debugging
        logger.error(
            f"Database error {error_id}: {error_type} - {str(error)[:500]}",
            extra={
                'error_id': error_id,
                'error_type': error_type,
                'request_path': request.url.path if request else None,
                'request_method': request.method if request else None,
                'client_ip': request.client.host if request else None
            },
            exc_info=True
        )
        
        # Determine appropriate response
        if error_type in cls.DB_ERROR_MAPPINGS:
            message, status_code = cls.DB_ERROR_MAPPINGS[error_type]
        else:
            message = "Database operation failed"
            status_code = 500
        
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": message,
                "error_id": error_id,
                "type": "database_error"
            }
        )
    
    @classmethod
    def handle_validation_error(cls, error: Union[ValidationError, RequestValidationError], request: Request = None) -> JSONResponse:
        """
        Handle validation errors securely
        
        Args:
            error: Validation exception
            request: Optional request object
            
        Returns:
            JSONResponse with sanitized validation errors
        """
        error_id = cls.generate_error_id()
        
        # Extract validation errors and sanitize
        if isinstance(error, RequestValidationError):
            errors = error.errors()
        elif isinstance(error, ValidationError):
            errors = error.errors()
        else:
            errors = [{"msg": "Validation failed"}]
        
        # Sanitize error messages to prevent information disclosure
        sanitized_errors = []
        for err in errors:
            sanitized_error = {
                "field": ".".join(str(loc) for loc in err.get("loc", [])) if err.get("loc") else "unknown",
                "message": cls._sanitize_validation_message(err.get("msg", "Invalid value")),
                "type": err.get("type", "validation_error")
            }
            sanitized_errors.append(sanitized_error)
        
        # Log the error for debugging
        logger.warning(
            f"Validation error {error_id}: {len(sanitized_errors)} validation issues",
            extra={
                'error_id': error_id,
                'error_count': len(sanitized_errors),
                'request_path': request.url.path if request else None,
                'validation_errors': sanitized_errors
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Validation failed",
                "errors": sanitized_errors,
                "error_id": error_id,
                "type": "validation_error"
            }
        )
    
    @classmethod
    def handle_http_exception(cls, error: HTTPException, request: Request = None) -> JSONResponse:
        """
        Handle HTTP exceptions securely
        
        Args:
            error: HTTP exception
            request: Optional request object
            
        Returns:
            JSONResponse with appropriate error message
        """
        error_id = cls.generate_error_id()
        status_code = error.status_code
        
        # Use generic message for security, unless it's a client error with safe detail
        if status_code in cls.GENERIC_MESSAGES:
            if status_code < 500:
                # For client errors (4xx), we can be more specific if the detail is safe
                detail = cls._sanitize_error_detail(error.detail)
            else:
                # For server errors (5xx), always use generic message
                detail = cls.GENERIC_MESSAGES[status_code]
        else:
            detail = cls.GENERIC_MESSAGES.get(500, "An error occurred")
        
        # Log appropriate level based on status code
        log_level = logging.ERROR if status_code >= 500 else logging.WARNING
        logger.log(
            log_level,
            f"HTTP error {error_id}: {status_code} - {error.detail}",
            extra={
                'error_id': error_id,
                'status_code': status_code,
                'original_detail': error.detail,
                'request_path': request.url.path if request else None,
                'request_method': request.method if request else None,
                'client_ip': request.client.host if request else None
            }
        )
        
        response_data = {
            "detail": detail,
            "error_id": error_id,
            "type": "http_error"
        }
        
        # Include headers if present
        headers = getattr(error, 'headers', None)
        
        return JSONResponse(
            status_code=status_code,
            content=response_data,
            headers=headers
        )
    
    @classmethod
    def handle_generic_exception(cls, error: Exception, request: Request = None) -> JSONResponse:
        """
        Handle unexpected exceptions securely
        
        Args:
            error: Generic exception
            request: Optional request object
            
        Returns:
            JSONResponse with generic error message
        """
        error_id = cls.generate_error_id()
        error_type = type(error).__name__
        
        # Log the full error for debugging
        logger.error(
            f"Unexpected error {error_id}: {error_type} - {str(error)[:500]}",
            extra={
                'error_id': error_id,
                'error_type': error_type,
                'request_path': request.url.path if request else None,
                'request_method': request.method if request else None,
                'client_ip': request.client.host if request else None
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": cls.GENERIC_MESSAGES[500],
                "error_id": error_id,
                "type": "internal_error"
            }
        )
    
    @classmethod
    def _sanitize_validation_message(cls, message: str) -> str:
        """Sanitize validation error messages"""
        # Common validation messages that are safe to show
        safe_messages = {
            "field required": "This field is required",
            "ensure this value has at least": "Value too short",
            "ensure this value has at most": "Value too long",
            "string too short": "Input too short",
            "string too long": "Input too long",
            "value is not a valid email": "Invalid email format",
            "value is not a valid integer": "Must be a number",
            "value is not a valid float": "Must be a valid decimal number",
            "invalid literal for int()": "Must be a valid number",
            "value is not a valid boolean": "Must be true or false",
            "extra fields not permitted": "Unknown field provided"
        }
        
        # Check for known safe patterns
        message_lower = message.lower()
        for pattern, safe_msg in safe_messages.items():
            if pattern in message_lower:
                return safe_msg
        
        # For unknown validation messages, return generic message
        return "Invalid input value"
    
    @classmethod
    def _sanitize_error_detail(cls, detail: Any) -> str:
        """Sanitize error detail to prevent information disclosure"""
        if isinstance(detail, str):
            # Check if it's a safe client error message
            detail_lower = detail.lower()
            
            # Safe patterns that can be shown to users
            safe_patterns = [
                "invalid", "required", "not found", "unauthorized", "forbidden",
                "expired", "too many", "limit", "format", "email", "password"
            ]
            
            if any(pattern in detail_lower for pattern in safe_patterns):
                return detail
        
        # For unsafe or non-string details, return generic message
        return "An error occurred"

def create_error_handlers(app):
    """
    Add secure error handlers to FastAPI application
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return SecurityErrorHandler.handle_validation_error(exc, request)
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        return SecurityErrorHandler.handle_validation_error(exc, request)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return SecurityErrorHandler.handle_http_exception(exc, request)
    
    @app.exception_handler(sqlalchemy.exc.IntegrityError)
    async def integrity_error_handler(request: Request, exc: sqlalchemy.exc.IntegrityError):
        return SecurityErrorHandler.handle_database_error(exc, request)
    
    @app.exception_handler(sqlalchemy.exc.OperationalError)
    async def operational_error_handler(request: Request, exc: sqlalchemy.exc.OperationalError):
        return SecurityErrorHandler.handle_database_error(exc, request)
    
    @app.exception_handler(sqlalchemy.exc.DatabaseError)
    async def database_error_handler(request: Request, exc: sqlalchemy.exc.DatabaseError):
        return SecurityErrorHandler.handle_database_error(exc, request)
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return SecurityErrorHandler.handle_generic_exception(exc, request)

class SecureResponseMiddleware:
    """Middleware to add security headers and sanitize responses"""
    
    def __init__(self, app, environment: str = "production"):
        self.app = app
        self.environment = environment
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Robots-Tag": "noindex, nofollow" if environment == "production" else None
        }
        
        # Add HSTS only in production with HTTPS
        if environment == "production":
            self.security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    
                    # Add security headers
                    for header, value in self.security_headers.items():
                        if value:  # Only add non-None headers
                            headers[header.encode()] = value.encode()
                    
                    # Remove server header to prevent server fingerprinting
                    headers.pop(b"server", None)
                    
                    message["headers"] = [(k.encode() if isinstance(k, str) else k, 
                                         v.encode() if isinstance(v, str) else v) 
                                        for k, v in headers.items()]
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

def setup_secure_error_handling(app, environment: str = "production"):
    """
    Setup comprehensive secure error handling for a FastAPI application
    
    Args:
        app: FastAPI application instance
        environment: Environment name (development, staging, production)
    """
    # Add error handlers
    create_error_handlers(app)
    
    # Add security response middleware
    app.add_middleware(SecureResponseMiddleware, environment=environment)
    
    logger.info(f"Secure error handling configured for {environment} environment") 