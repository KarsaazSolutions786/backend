"""
Standardized Authentication Integration for Eindr Microservices

This module provides a consistent authentication interface for all microservices,
integrating with the shared security modules.
"""

import os
import logging
from typing import Dict, Optional, List, Union
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

# Import shared security modules
from .auth_utils import SecureJWTValidator, jwt_validator, get_current_user, get_current_user_payload
from .rbac import PermissionService, Permission, require_permission
from .rate_limiting import RateLimitService
from .input_validation import validate_input
from .secure_error_handling import handle_auth_error

logger = logging.getLogger(__name__)

class ServiceAuthConfig:
    """Authentication configuration for microservices"""
    
    # Service identification
    SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown-service")
    
    # Auth service URL for HTTP-based validation (fallback)
    AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8000")
    
    # JWT Configuration
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    
    # Authentication mode: 'jwt' for direct validation, 'service' for HTTP calls
    AUTH_MODE = os.getenv("AUTH_MODE", "jwt")
    
    # Service-specific permissions
    REQUIRED_PERMISSIONS = {
        "chat-service": [Permission.USE_CHAT_AI, Permission.CREATE_CONVERSATION],
        "note-service": [Permission.CREATE_NOTE, Permission.READ_NOTE],
        "reminder-service": [Permission.CREATE_REMINDER, Permission.READ_REMINDER],
        "ledger-service": [Permission.CREATE_LEDGER_ENTRY, Permission.READ_LEDGER_ENTRY],
        "customer-service": [Permission.READ_CUSTOMER, Permission.UPDATE_CUSTOMER],
        "intent-service": [Permission.USE_INTENT_CLASSIFICATION],
        "stt-service": [Permission.USE_STT],
        "tts-service": [Permission.USE_TTS],
        "ai-pipeline-service": [Permission.USE_AI_PIPELINE],
    }

class ServiceAuthManager:
    """Centralized authentication manager for microservices"""
    
    def __init__(self, config: ServiceAuthConfig = None):
        self.config = config or ServiceAuthConfig()
        self.jwt_validator = jwt_validator
        self.security = HTTPBearer()
        
        # Validate configuration
        if not self.config.SECRET_KEY:
            raise ValueError("SECRET_KEY must be configured for JWT authentication")
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        db: Optional[Session] = None
    ) -> Dict:
        """
        Get current authenticated customer with comprehensive validation
        
        Args:
            credentials: JWT credentials from request
            db: Database session for RBAC (optional)
            
        Returns:
            User information dictionary
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Extract token
            token = credentials.credentials
            
            # Validate JWT token using shared auth utils
            payload = jwt_validator.verify_token(token, "access")
            customer_id = jwt_validator.get_customer_id_from_token(token)
            user_data = {
                "customer_id": customer_id,
                "email": payload.get("email", ""),
                "is_verified": payload.get("is_verified", True),
                "is_active": payload.get("is_active", True)
            }
            
            # Additional service-specific validation
            if db and self.config.SERVICE_NAME in self.config.REQUIRED_PERMISSIONS:
                permission_service = PermissionService(db)
                required_perms = self.config.REQUIRED_PERMISSIONS[self.config.SERVICE_NAME]
                
                customer_id = user_data.get("customer_id")
                if customer_id and required_perms:
                    # Check if user has required permissions for this service
                    for perm in required_perms:
                        if not permission_service.has_permission(customer_id, perm):
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"Insufficient permissions for {self.config.SERVICE_NAME}"
                            )
            
            logger.info(f"Authenticated user {user_data.get('customer_id')} for {self.config.SERVICE_NAME}")
            return user_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error in {self.config.SERVICE_NAME}: {e}")
            raise handle_auth_error(e)
    
    async def get_optional_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        db: Optional[Session] = None
    ) -> Optional[Dict]:
        """
        Get current customer if authenticated, otherwise return None
        
        Args:
            credentials: Optional JWT credentials
            db: Database session (optional)
            
        Returns:
            User data or None if not authenticated
        """
        if not credentials:
            return None
        
        try:
            return await self.get_current_user(credentials, db)
        except HTTPException:
            return None
    
    def require_auth(self, permissions: List[Union[str, Permission]] = None):
        """
        Dependency factory for requiring authentication with optional permissions
        
        Args:
            permissions: List of required permissions
            
        Returns:
            FastAPI dependency function
        """
        async def auth_dependency(
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
            db: Optional[Session] = None
        ) -> Dict:
            user = await self.get_current_user(credentials, db)
            
            # Check additional permissions if specified
            if permissions and db:
                permission_service = PermissionService(db)
                customer_id = user.get("customer_id")
                
                for perm in permissions:
                    if not permission_service.has_permission(customer_id, perm):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Insufficient permissions"
                        )
            
            return user
        
        return auth_dependency
    
    def create_service_headers(self, customer_id: int) -> Dict[str, str]:
        """
        Create authentication headers for service-to-service communication
        
        Args:
            customer_id: Customer ID to include in token
            
        Returns:
            Headers dictionary with Authorization header
        """
        # For service-to-service communication, return headers without token for now
        # In production, implement proper service tokens
        return {}

class LegacyJWTMigration:
    """Helper for migrating from legacy JWT implementations"""
    
    @staticmethod
    def detect_legacy_pattern(file_content: str) -> bool:
        """Detect if file uses legacy JWT pattern"""
        legacy_indicators = [
            "from jose import jwt",
            'SECRET_KEY = os.getenv("JWT_SECRET"',
            "supersecret",
            "jwt.decode(token, SECRET_KEY",
        ]
        return any(indicator in file_content for indicator in legacy_indicators)
    
    @staticmethod
    def get_migration_instructions(service_name: str) -> str:
        """Get migration instructions for a service"""
        return f"""
# Migration Instructions for {service_name}

1. Replace jwt.py imports:
   - Remove: from services.{service_name}.src.utils.jwt import get_current_user
   - Add: from shared.service_auth import service_auth_manager

2. Update route dependencies:
   - Old: customer_id = Depends(get_current_user)
   - New: current_customer = Depends(service_auth_manager.require_auth())

3. Update route handlers:
   - Old: def endpoint(customer_id: int):
   - New: def endpoint(current_customer: Dict):
   - Access customer ID: current_customer["customer_id"]

4. Add to main.py:
   from shared.service_auth import service_auth_manager
   from shared.security_config import get_cors_settings, SecurityHeadersMiddleware
   from shared.rate_limiting import RateLimitMiddleware

5. Configure middleware:
   app.add_middleware(SecurityHeadersMiddleware)
   app.add_middleware(RateLimitMiddleware)
"""

# Global service auth manager instance
service_auth_manager = ServiceAuthManager()

# Convenience dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Optional[Session] = None
) -> Dict:
    """Get current authenticated customer"""
    return await service_auth_manager.get_current_user(credentials, db)

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Optional[Session] = None
) -> Optional[Dict]:
    """Get current customer if authenticated"""
    return await service_auth_manager.get_optional_user(credentials, db)

def require_auth(permissions: List[Union[str, Permission]] = None):
    """Require authentication with optional permissions"""
    return service_auth_manager.require_auth(permissions)

def create_service_headers(customer_id: int) -> Dict[str, str]:
    """Create headers for service-to-service communication"""
    return service_auth_manager.create_service_headers(customer_id)

# Service-specific permission requirements
def require_chat_permissions():
    """Require permissions for chat service"""
    return require_auth([Permission.USE_CHAT_AI, Permission.CREATE_CONVERSATION])

def require_note_permissions():
    """Require permissions for note service"""
    return require_auth([Permission.CREATE_NOTE, Permission.READ_NOTE])

def require_reminder_permissions():
    """Require permissions for reminder service"""
    return require_auth([Permission.CREATE_REMINDER, Permission.READ_REMINDER])

def require_ledger_permissions():
    """Require permissions for ledger service"""
    return require_auth([Permission.CREATE_LEDGER_ENTRY, Permission.READ_LEDGER_ENTRY])

def require_ai_permissions():
    """Require permissions for AI services"""
    return require_auth([Permission.USE_STT, Permission.USE_TTS, Permission.USE_INTENT_CLASSIFICATION])

def require_customer_permissions():
    """Require permissions for customer service"""
    return require_auth([Permission.READ_CUSTOMER, Permission.UPDATE_CUSTOMER])

# Backward compatibility functions (to ease migration)
def get_current_user_legacy(request: Request) -> int:
    """
    Legacy compatibility function - DO NOT USE IN NEW CODE
    This exists only to ease migration from old jwt.py files
    """
    logger.warning("Using legacy authentication function - please migrate to new service_auth module")
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ")[1]
    
    try:
        # Use the new secure JWT validation
        user_data = verify_jwt_token(token)
        return user_data.get("customer_id")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token") 