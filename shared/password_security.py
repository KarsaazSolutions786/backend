"""
Comprehensive Password Security for Eindr Microservices

This module provides secure password handling, strength validation, and
related security utilities for user authentication.
"""

import bcrypt
import secrets
import hashlib
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from zxcvbn import zxcvbn
import os

logger = logging.getLogger(__name__)

class PasswordSecurityConfig:
    """Configuration for password security requirements"""
    
    # Password requirements
    MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
    MAX_LENGTH = int(os.getenv("PASSWORD_MAX_LENGTH", "128"))
    REQUIRE_UPPERCASE = os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true"
    REQUIRE_LOWERCASE = os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true"
    REQUIRE_DIGITS = os.getenv("PASSWORD_REQUIRE_DIGITS", "true").lower() == "true"
    REQUIRE_SPECIAL = os.getenv("PASSWORD_REQUIRE_SPECIAL", "true").lower() == "true"
    MIN_STRENGTH_SCORE = int(os.getenv("PASSWORD_MIN_STRENGTH_SCORE", "3"))  # 0-4 scale
    
    # Bcrypt configuration
    BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))
    
    # Common passwords and patterns to reject
    COMMON_PASSWORDS = {
        "password", "123456", "password123", "admin", "qwerty", "letmein",
        "welcome", "monkey", "dragon", "111111", "123123", "1234567890",
        "sunshine", "princess", "football", "baseball", "welcome123"
    }
    
    # Common patterns to reject
    WEAK_PATTERNS = [
        r"^(.)\1+$",  # All same character
        r"^(012|123|234|345|456|567|678|789|890)+",  # Sequential numbers
        r"^(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)+",  # Sequential letters
        r"^(qwe|asd|zxc)",  # Keyboard patterns
    ]

class SecurePasswordManager:
    """Secure password hashing and validation manager"""
    
    def __init__(self, config: PasswordSecurityConfig = None):
        self.config = config or PasswordSecurityConfig()
    
    def hash_password(self, password: str) -> str:
        """
        Securely hash a password using bcrypt
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Bcrypt hashed password string
            
        Raises:
            ValueError: If password is invalid
        """
        if not isinstance(password, str):
            raise ValueError("Password must be a string")
        
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Validate password strength before hashing
        validation_result = self.validate_password_strength(password)
        if not validation_result["valid"]:
            raise ValueError(f"Password does not meet security requirements: {validation_result['errors']}")
        
        # Hash with bcrypt
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt(rounds=self.config.BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            password: Plain text password to verify
            hashed_password: Bcrypt hashed password
            
        Returns:
            True if password matches hash, False otherwise
        """
        if not password or not hashed_password:
            return False
        
        try:
            password_bytes = password.encode('utf-8')
            hashed_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception as e:
            logger.warning(f"Password verification error: {e}")
            return False
    
    def validate_password_strength(self, password: str, user_info: Dict = None) -> Dict:
        """
        Comprehensive password strength validation
        
        Args:
            password: Password to validate
            user_info: Optional user information (email, name) to check against
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        score = 0
        
        # Basic length validation
        if len(password) < self.config.MIN_LENGTH:
            errors.append(f"Password must be at least {self.config.MIN_LENGTH} characters long")
        
        if len(password) > self.config.MAX_LENGTH:
            errors.append(f"Password must be no more than {self.config.MAX_LENGTH} characters long")
        
        # Character requirements
        if self.config.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.REQUIRE_DIGITS and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if self.config.REQUIRE_SPECIAL and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check against common passwords
        if password.lower() in self.config.COMMON_PASSWORDS:
            errors.append("Password is too common")
        
        # Check against weak patterns
        for pattern in self.config.WEAK_PATTERNS:
            if re.search(pattern, password.lower()):
                errors.append("Password contains a weak pattern")
                break
        
        # Check against user information
        if user_info:
            email = user_info.get("email", "")
            name = user_info.get("name", "")
            
            if email and email.split("@")[0].lower() in password.lower():
                errors.append("Password should not contain your email username")
            
            if name and len(name) > 2 and name.lower() in password.lower():
                errors.append("Password should not contain your name")
        
        # Use zxcvbn for advanced strength checking
        try:
            user_inputs = []
            if user_info:
                user_inputs = [user_info.get("email", ""), user_info.get("name", "")]
            
            zxcvbn_result = zxcvbn(password, user_inputs=user_inputs)
            score = zxcvbn_result["score"]
            
            if score < self.config.MIN_STRENGTH_SCORE:
                errors.append(f"Password is too weak (strength: {score}/4, required: {self.config.MIN_STRENGTH_SCORE}/4)")
                
                # Add specific feedback from zxcvbn
                feedback = zxcvbn_result.get("feedback", {})
                suggestions = feedback.get("suggestions", [])
                if suggestions:
                    errors.extend(suggestions)
        
        except ImportError:
            logger.warning("zxcvbn not available, using basic strength validation")
            # Fallback strength calculation
            if len(password) >= 12:
                score += 1
            if re.search(r'[A-Z]', password) and re.search(r'[a-z]', password):
                score += 1
            if re.search(r'\d', password):
                score += 1
            if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                score += 1
            
            if score < self.config.MIN_STRENGTH_SCORE:
                errors.append(f"Password is too weak (score: {score}/4)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "strength_score": score,
            "estimated_crack_time": getattr(zxcvbn_result, "crack_times_display", {}).get("offline_slow_hashing_1e4_per_second", "Unknown") if 'zxcvbn_result' in locals() else "Unknown"
        }
    
    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a cryptographically secure password
        
        Args:
            length: Length of password to generate (minimum 12)
            
        Returns:
            Secure random password
        """
        if length < 12:
            length = 12
        
        # Character sets
        lowercase = "abcdefghijklmnopqrstuvwxyz"
        uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        special = "!@#$%^&*(),.?\":{}|<>"
        
        # Ensure at least one character from each required set
        password_chars = []
        
        if self.config.REQUIRE_LOWERCASE:
            password_chars.append(secrets.choice(lowercase))
        if self.config.REQUIRE_UPPERCASE:
            password_chars.append(secrets.choice(uppercase))
        if self.config.REQUIRE_DIGITS:
            password_chars.append(secrets.choice(digits))
        if self.config.REQUIRE_SPECIAL:
            password_chars.append(secrets.choice(special))
        
        # Fill remaining length with random characters from all sets
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - len(password_chars)):
            password_chars.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password_chars)
        
        return ''.join(password_chars)

class SecureTokenManager:
    """Manager for secure token generation and validation"""
    
    @staticmethod
    def generate_reset_token() -> str:
        """Generate secure password reset token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_verification_token() -> str:
        """Generate secure email verification token"""
        return secrets.token_urlsafe(24)
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Hash a token for secure storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    @staticmethod
    def verify_token_hash(token: str, token_hash: str) -> bool:
        """Verify token against its hash"""
        return hashlib.sha256(token.encode()).hexdigest() == token_hash

class PasswordPolicyEnforcer:
    """Enforce password policies and track password history"""
    
    def __init__(self, password_manager: SecurePasswordManager = None):
        self.password_manager = password_manager or SecurePasswordManager()
        self.password_history_limit = int(os.getenv("PASSWORD_HISTORY_LIMIT", "5"))
        self.password_max_age_days = int(os.getenv("PASSWORD_MAX_AGE_DAYS", "90"))
    
    def can_reuse_password(self, new_password: str, password_history: List[str]) -> bool:
        """
        Check if password can be reused based on history
        
        Args:
            new_password: New password to check
            password_history: List of previous password hashes
            
        Returns:
            True if password can be used, False if it's been used recently
        """
        for old_hash in password_history[-self.password_history_limit:]:
            if self.password_manager.verify_password(new_password, old_hash):
                return False
        return True
    
    def password_expired(self, last_changed: datetime) -> bool:
        """
        Check if password has expired
        
        Args:
            last_changed: Date when password was last changed
            
        Returns:
            True if password has expired
        """
        if self.password_max_age_days <= 0:
            return False  # No expiration policy
        
        expiry_date = last_changed + timedelta(days=self.password_max_age_days)
        return datetime.utcnow() > expiry_date
    
    def days_until_expiry(self, last_changed: datetime) -> int:
        """
        Calculate days until password expires
        
        Args:
            last_changed: Date when password was last changed
            
        Returns:
            Number of days until expiry (negative if already expired)
        """
        if self.password_max_age_days <= 0:
            return -1  # No expiration policy
        
        expiry_date = last_changed + timedelta(days=self.password_max_age_days)
        delta = expiry_date - datetime.utcnow()
        return delta.days

# Global instances for easy access
default_password_manager = SecurePasswordManager()
default_token_manager = SecureTokenManager()
default_policy_enforcer = PasswordPolicyEnforcer(default_password_manager)

# Convenience functions
def hash_password(password: str) -> str:
    """Hash a password using default configuration"""
    return default_password_manager.hash_password(password)

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password using default configuration"""
    return default_password_manager.verify_password(password, hashed_password)

def validate_password_strength(password: str, user_info: Dict = None) -> Dict:
    """Validate password strength using default configuration"""
    return default_password_manager.validate_password_strength(password, user_info)

def generate_secure_password(length: int = 16) -> str:
    """Generate a secure password using default configuration"""
    return default_password_manager.generate_secure_password(length) 