"""
Comprehensive Input Validation and Sanitization for Eindr Microservices

This module provides secure input validation, sanitization, and Pydantic models
to prevent injection attacks and ensure data integrity.
"""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from fastapi import HTTPException, status
import bleach
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class SecurityError(HTTPException):
    """Custom exception for security-related validation failures"""
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code=status_code, detail=detail)

class InputSanitizer:
    """Comprehensive input sanitization utilities"""
    
    # Patterns for common injection attacks
    SQL_INJECTION_PATTERNS = [
        r"('|(\\'))|(;)|(\s(or|and)\s+[\w'\"]+\s*=)", r"(union|select|insert|update|delete|drop|create|alter|exec|execute)",
        r"(<script|<iframe|<object|<embed|<link|<meta)", r"(javascript:|vbscript:|onload=|onerror=|onclick=)"
    ]
    
    XSS_PATTERNS = [
        r"<script.*?>.*?</script>", r"javascript:", r"vbscript:", r"onload\s*=", r"onerror\s*=",
        r"onclick\s*=", r"onmouseover\s*=", r"<iframe", r"<object", r"<embed"
    ]
    
    # Safe HTML tags for rich text (if needed)
    ALLOWED_HTML_TAGS = ["b", "i", "u", "em", "strong", "p", "br", "ul", "ol", "li"]
    ALLOWED_HTML_ATTRIBUTES = {"a": ["href"], "p": ["class"], "span": ["class"]}
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """
        Sanitize string input to prevent injection attacks
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow safe HTML tags
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If potentially malicious content is detected
        """
        if not isinstance(value, str):
            return str(value)
        
        # Check length
        if len(value) > max_length:
            raise SecurityError(f"Input too long. Maximum {max_length} characters allowed.")
        
        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential SQL injection attempt detected: {pattern}")
                raise SecurityError("Invalid input detected. Please check your input and try again.")
        
        # Check for XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential XSS attempt detected: {pattern}")
                raise SecurityError("Invalid input detected. Please check your input and try again.")
        
        # Sanitize HTML if not allowed
        if allow_html:
            # Use bleach to clean HTML, allowing only safe tags
            value = bleach.clean(
                value,
                tags=cls.ALLOWED_HTML_TAGS,
                attributes=cls.ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
        else:
            # Escape HTML entities
            value = html.escape(value)
        
        # Remove any null bytes
        value = value.replace('\x00', '')
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value.strip())
        
        return value
    
    @classmethod
    def sanitize_email(cls, email: str) -> str:
        """
        Sanitize and validate email address
        
        Args:
            email: Email address to validate
            
        Returns:
            Normalized email address
            
        Raises:
            SecurityError: If email is invalid
        """
        if not email or not isinstance(email, str):
            raise SecurityError("Email address is required")
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        # Normalize email
        email = email.lower().strip()
        
        # Validate format
        if not re.match(email_pattern, email):
            raise SecurityError("Invalid email address format")
        
        # Check length
        if len(email) > 254:  # RFC 5321 limit
            raise SecurityError("Email address too long")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'[<>"\']',  # HTML/SQL injection chars
            r'javascript:',  # XSS
            r'\s',  # Whitespace in email
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, email):
                raise SecurityError("Invalid characters in email address")
        
        return email
    
    @classmethod
    def sanitize_url(cls, url: str, allowed_schemes: List[str] = ["http", "https"]) -> str:
        """
        Sanitize and validate URL
        
        Args:
            url: URL to validate
            allowed_schemes: List of allowed URL schemes
            
        Returns:
            Validated URL
            
        Raises:
            SecurityError: If URL is invalid or uses disallowed scheme
        """
        if not url or not isinstance(url, str):
            raise SecurityError("URL is required")
        
        url = url.strip()
        
        try:
            parsed = urlparse(url)
        except Exception:
            raise SecurityError("Invalid URL format")
        
        # Check scheme
        if parsed.scheme.lower() not in [s.lower() for s in allowed_schemes]:
            raise SecurityError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
        
        # Check for suspicious patterns
        if re.search(r'javascript:|vbscript:|data:', url, re.IGNORECASE):
            raise SecurityError("Potentially unsafe URL detected")
        
        return url
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and other attacks
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Safe filename
            
        Raises:
            SecurityError: If filename contains suspicious patterns
        """
        if not filename or not isinstance(filename, str):
            raise SecurityError("Filename is required")
        
        # Remove path separators and relative path indicators
        filename = filename.replace('/', '').replace('\\', '').replace('..', '')
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            raise SecurityError("Filename too long")
        
        # Ensure filename is not empty after sanitization
        if not filename.strip():
            raise SecurityError("Invalid filename")
        
        return filename.strip()

class SecureBaseModel(BaseModel):
    """Base Pydantic model with automatic input sanitization"""
    
    class Config:
        # Validate assignment
        validate_assignment = True
        # Allow population by field name
        validate_by_name = True
        # Forbid extra fields
        extra = "forbid"
    
    @root_validator(pre=True)
    def sanitize_inputs(cls, values):
        """Automatically sanitize string inputs"""
        if isinstance(values, dict):
            sanitized = {}
            for key, value in values.items():
                if isinstance(value, str):
                    try:
                        sanitized[key] = InputSanitizer.sanitize_string(value)
                    except SecurityError:
                        # Re-raise with field context
                        raise SecurityError(f"Invalid input in field '{key}': Please check your input and try again.")
                else:
                    sanitized[key] = value
            return sanitized
        return values

# Common secure field validators
def secure_string_field(max_length: int = 255, min_length: int = 1, allow_html: bool = False):
    """Create a secure string field with validation"""
    return Field(
        ...,
        min_length=min_length,
        max_length=max_length,
        description=f"String field (length: {min_length}-{max_length})"
    )

def secure_email_field():
    """Create a secure email field with validation"""
    return Field(
        ...,
        pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        max_length=254,
        description="Valid email address"
    )

def secure_url_field():
    """Create a secure URL field with validation"""
    return Field(
        ...,
        pattern=r'^https?://[^\s/$.?#].[^\s]*$',
        max_length=2048,
        description="Valid HTTP/HTTPS URL"
    )

# Common secure Pydantic models
class SecureUserRegistration(SecureBaseModel):
    """Secure user registration model"""
    email: str = secure_email_field()
    password: str = Field(..., min_length=8, max_length=128, description="Password (8-128 characters)")
    
    @validator('email')
    def validate_email(cls, v):
        return InputSanitizer.sanitize_email(v)
    
    @validator('password')
    def validate_password(cls, v):
        # Basic password requirements
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'\d', v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError("Password must contain at least one special character")
        return v

class SecureUserLogin(SecureBaseModel):
    """Secure user login model"""
    email: str = secure_email_field()
    password: str = Field(..., min_length=1, max_length=128)
    remember_me: bool = Field(default=False)
    
    @validator('email')
    def validate_email(cls, v):
        return InputSanitizer.sanitize_email(v)

class SecureTextContent(SecureBaseModel):
    """Secure text content model for notes, messages, etc."""
    title: str = secure_string_field(max_length=200)
    content: str = secure_string_field(max_length=10000, allow_html=False)
    tags: Optional[List[str]] = Field(default=[], max_items=10)
    
    @validator('tags')
    def validate_tags(cls, v):
        if not v:
            return []
        sanitized_tags = []
        for tag in v:
            if not isinstance(tag, str):
                continue
            sanitized_tag = InputSanitizer.sanitize_string(tag, max_length=50)
            if sanitized_tag:
                sanitized_tags.append(sanitized_tag)
        return sanitized_tags[:10]  # Limit to 10 tags

class SecureFinancialTransaction(SecureBaseModel):
    """Secure financial transaction model"""
    amount: float = Field(..., gt=0, le=1000000, description="Transaction amount (positive, max 1M)")
    description: str = secure_string_field(max_length=500)
    category: Optional[str] = secure_string_field(max_length=100, min_length=0)
    
    @validator('amount')
    def validate_amount(cls, v):
        # Round to 2 decimal places for currency
        if v <= 0:
            raise ValueError("Amount must be positive")
        return round(float(v), 2)

class ValidationUtils:
    """Additional validation utilities"""
    
    @staticmethod
    def validate_pagination(page: int = 1, size: int = 20, max_size: int = 100) -> tuple:
        """
        Validate pagination parameters
        
        Args:
            page: Page number (1-based)
            size: Page size
            max_size: Maximum allowed page size
            
        Returns:
            Tuple of (validated_page, validated_size)
        """
        if page < 1:
            page = 1
        if size < 1:
            size = 20
        if size > max_size:
            size = max_size
        
        return page, size
    
    @staticmethod
    def validate_sort_field(field: str, allowed_fields: List[str]) -> str:
        """
        Validate sort field against allowed list
        
        Args:
            field: Field name to sort by
            allowed_fields: List of allowed field names
            
        Returns:
            Validated field name
            
        Raises:
            SecurityError: If field is not in allowed list
        """
        if field not in allowed_fields:
            raise SecurityError(f"Invalid sort field. Allowed fields: {', '.join(allowed_fields)}")
        return field
    
    @staticmethod
    def sanitize_search_query(query: str, max_length: int = 100) -> str:
        """
        Sanitize search query to prevent injection
        
        Args:
            query: Search query string
            max_length: Maximum query length
            
        Returns:
            Sanitized query string
        """
        if not query:
            return ""
        
        # Remove special regex characters that could cause issues
        query = re.sub(r'[.*+?^${}()|[\]\\]', '', query)
        
        # Sanitize as regular string
        query = InputSanitizer.sanitize_string(query, max_length=max_length)
        
        return query.strip()