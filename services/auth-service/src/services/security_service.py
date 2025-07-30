from fastapi import HTTPException, status, Request
from typing import Tuple, Optional
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Check if enhanced security modules are available
try:
    from rate_limiting import brute_force_protection, record_auth_failure
    from input_validation import SecureUserRegistration, SecureUserLogin, InputSanitizer, SecurityError
    from security_config import SensitiveDataFilter
    HAS_ENHANCED_SECURITY = True
    logger.info("Enhanced security modules loaded successfully")
except ImportError:
    HAS_ENHANCED_SECURITY = False
    logger.warning("Enhanced security modules not available, using basic security")

class SecurityService:
    """Service for handling security operations"""
    
    def __init__(self):
        self.has_enhanced_security = HAS_ENHANCED_SECURITY
        self._init_basic_security()
    
    def _init_basic_security(self):
        """Initialize basic security patterns and rules"""
        # Email validation pattern
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # Password requirements
        self.min_password_length = 8
        self.max_password_length = 128
        
        # Rate limiting (basic implementation)
        self.rate_limit_attempts = {}
        self.max_attempts_per_minute = 5
        self.lockout_duration_minutes = 15
    
    def validate_request_security(self, request: Request, endpoint: str):
        """Validate request for security (rate limiting, etc.)"""
        if self.has_enhanced_security:
            try:
                brute_force_protection.check_rate_limit(request, endpoint)
            except HTTPException as e:
                logger.warning(f"Rate limit exceeded for {endpoint}: {e.detail}")
                raise
        else:
            # Basic rate limiting
            self._basic_rate_limit_check(request, endpoint)
    
    def _basic_rate_limit_check(self, request: Request, endpoint: str):
        """Basic rate limiting implementation"""
        client_ip = request.client.host
        current_time = datetime.utcnow()
        key = f"{client_ip}:{endpoint}"
        
        # Clean old attempts
        cutoff_time = current_time - timedelta(minutes=1)
        if key in self.rate_limit_attempts:
            self.rate_limit_attempts[key] = [
                attempt_time for attempt_time in self.rate_limit_attempts[key]
                if attempt_time > cutoff_time
            ]
        
        # Check current attempts
        attempts = self.rate_limit_attempts.get(key, [])
        if len(attempts) >= self.max_attempts_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )
        
        # Record this attempt
        if key not in self.rate_limit_attempts:
            self.rate_limit_attempts[key] = []
        self.rate_limit_attempts[key].append(current_time)
    
    def validate_email(self, email: str) -> str:
        """Validate and sanitize email address"""
        if not email or not isinstance(email, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required"
            )
        
        email = email.strip().lower()
        
        if len(email) > 254:  # RFC 5321 limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address is too long"
            )
        
        if not self.email_pattern.match(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        return email
    
    def validate_password(self, password: str) -> str:
        """Validate password strength"""
        if not password or not isinstance(password, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is required"
            )
        
        if len(password) < self.min_password_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password must be at least {self.min_password_length} characters long"
            )
        
        if len(password) > self.max_password_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password must be no more than {self.max_password_length} characters long"
            )
        
        # Check for basic password requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        if not (has_upper and has_lower and has_digit):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one uppercase letter, one lowercase letter, and one digit"
            )
        
        return password
    
    def validate_registration_data(self, email: str, password: str, confirm_password: str) -> Tuple[str, str]:
        """Validate registration data with enhanced security if available"""
        if self.has_enhanced_security:
            try:
                secure_data = SecureUserRegistration(
                    email=email,
                    password=password
                )
                return secure_data.email, secure_data.password
            except (ValueError, SecurityError) as e:
                logger.warning(f"Enhanced security validation failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid registration data. Please check your input."
                )
        else:
            # Basic validation
            if password != confirm_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Passwords do not match"
                )
            
            validated_email = self.validate_email(email)
            validated_password = self.validate_password(password)
            
            return validated_email, validated_password
    
    def validate_login_data(self, email: str, password: str, request: Request, remember_me: bool = False) -> Tuple[str, str, bool]:
        """Validate login data with enhanced security if available"""
        if self.has_enhanced_security:
            try:
                secure_data = SecureUserLogin(
                    email=email,
                    password=password,
                    remember_me=remember_me
                )
                return secure_data.email, secure_data.password, secure_data.remember_me
            except (ValueError, SecurityError) as e:
                logger.warning(f"Enhanced security validation failed: {e}")
                self.record_auth_failure(request, "auth.login")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid login data"
                )
        else:
            # Basic validation
            validated_email = self.validate_email(email)
            
            if not password or not isinstance(password, str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password is required"
                )
            
            return validated_email, password, remember_me
    
    def record_auth_failure(self, request: Request, endpoint: str):
        """Record authentication failure for security monitoring"""
        if self.has_enhanced_security:
            try:
                record_auth_failure(request, endpoint)
            except Exception as e:
                logger.error(f"Failed to record auth failure: {e}")
        else:
            # Basic logging
            client_ip = request.client.host
            user_agent = request.headers.get("user-agent", "unknown")
            logger.warning(f"Auth failure recorded - IP: {client_ip}, Endpoint: {endpoint}, User-Agent: {user_agent}")
    
    def sanitize_input(self, input_data: str, max_length: int = 1000) -> str:
        """Sanitize user input"""
        if self.has_enhanced_security:
            try:
                sanitizer = InputSanitizer()
                return sanitizer.sanitize(input_data, max_length=max_length)
            except Exception as e:
                logger.warning(f"Enhanced input sanitization failed: {e}")
                # Fall back to basic sanitization
        
        # Basic sanitization
        if not input_data or not isinstance(input_data, str):
            return ""
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in input_data if ord(char) >= 32 or char in '\t\n\r')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def filter_sensitive_data(self, data: dict) -> dict:
        """Filter sensitive data from logs/responses"""
        if self.has_enhanced_security:
            try:
                filter_service = SensitiveDataFilter()
                return filter_service.filter_dict(data)
            except Exception as e:
                logger.warning(f"Enhanced data filtering failed: {e}")
                # Fall back to basic filtering
        
        # Basic sensitive data filtering
        sensitive_keys = {
            'password', 'confirm_password', 'token', 'refresh_token', 
            'access_token', 'secret', 'key', 'auth', 'authorization'
        }
        
        filtered_data = {}
        for key, value in data.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                filtered_data[key] = "[FILTERED]"
            elif isinstance(value, dict):
                filtered_data[key] = self.filter_sensitive_data(value)
            else:
                filtered_data[key] = value
        
        return filtered_data
    
    def validate_device_id(self, device_id: Optional[str]) -> Optional[str]:
        """Validate and sanitize device ID"""
        if not device_id:
            return None
        
        # Basic validation - alphanumeric and hyphens only
        if not re.match(r'^[a-zA-Z0-9-_]{1,64}$', device_id):
            logger.warning(f"Invalid device ID format: {device_id}")
            return None
        
        return device_id
    
    def validate_user_agent(self, user_agent: Optional[str]) -> Optional[str]:
        """Validate and sanitize user agent"""
        if not user_agent:
            return None
        
        # Sanitize and truncate user agent
        sanitized = self.sanitize_input(user_agent, max_length=500)
        
        # Basic validation - should contain some expected patterns
        if len(sanitized) < 5 or not any(pattern in sanitized.lower() for pattern in ['mozilla', 'chrome', 'safari', 'firefox', 'edge', 'opera', 'mobile', 'app']):
            logger.warning(f"Suspicious user agent: {sanitized}")
        
        return sanitized
    
    def is_security_enhanced(self) -> bool:
        """Check if enhanced security is available"""
        return self.has_enhanced_security
    
    def get_security_status(self) -> dict:
        """Get current security configuration status"""
        return {
            "enhanced_security_available": self.has_enhanced_security,
            "rate_limiting_enabled": True,
            "input_validation_enabled": True,
            "password_requirements": {
                "min_length": self.min_password_length,
                "max_length": self.max_password_length,
                "requires_uppercase": True,
                "requires_lowercase": True,
                "requires_digit": True
            },
            "max_attempts_per_minute": self.max_attempts_per_minute,
            "lockout_duration_minutes": self.lockout_duration_minutes
        }