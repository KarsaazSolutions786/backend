"""
Robust Shared Module Importer for Eindr Auth Service

This module provides a reliable way to import shared modules across different
deployment environments (local, Docker, Railway, etc.)
"""

import sys
import os
import importlib.util
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SharedModuleImporter:
    """Handles robust importing of shared modules with fallback mechanisms"""
    
    def __init__(self):
        self.shared_loaded = False
        self.refresh_token_service = None
        self.refresh_token_base = None
        self._setup_import_paths()
    
    def _setup_import_paths(self):
        """Configure import paths for different deployment environments"""
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        
        # Debug: Print current environment
        logger.info(f"Current file: {current_file}")
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Current sys.path: {sys.path}")
        
        # Define possible shared module locations
        possible_paths = [
            '/app/shared',  # Docker/Railway deployment
            os.path.join(current_dir, '..', '..', '..', 'shared'),  # Local development
            os.path.join(current_dir, '..', '..', 'shared'),  # Alternative local
            os.path.join(os.getcwd(), 'shared'),  # Current working directory
            os.path.join(os.path.dirname(os.getcwd()), 'shared'),  # Parent directory
        ]
        
        # Debug: Check all possible paths
        logger.info("Checking possible shared module paths:")
        for path in possible_paths:
            exists = os.path.exists(path)
            logger.info(f"  {path} - {'EXISTS' if exists else 'NOT FOUND'}")
            if exists:
                try:
                    files = os.listdir(path)
                    logger.info(f"    Contents: {files}")
                except Exception as e:
                    logger.error(f"    Error listing contents: {e}")
        
        # Add valid paths to sys.path
        for path in possible_paths:
            if path not in sys.path and os.path.exists(path):
                sys.path.insert(0, path)
                logger.info(f"Added shared path: {path}")
    
    def _import_shared_module(self) -> bool:
        """Attempt to import the shared module"""
        try:
            import shared
            logger.info("Successfully imported shared module")
            logger.info(f"Shared module file: {shared.__file__ if hasattr(shared, '__file__') else 'Unknown'}")
            logger.info(f"Shared module version: {shared.__version__ if hasattr(shared, '__version__') else 'Unknown'}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import shared module: {e}")
            logger.error(f"Import error details: {type(e).__name__}: {str(e)}")
            
            # Try to provide more debugging information
            try:
                import sys
                logger.error(f"Current sys.path when import failed: {sys.path}")
                
                # Check if shared directory exists in any sys.path location
                for path in sys.path:
                    shared_path = os.path.join(path, 'shared')
                    if os.path.exists(shared_path):
                        logger.error(f"Found shared directory at: {shared_path}")
                        try:
                            files = os.listdir(shared_path)
                            logger.error(f"  Contents: {files}")
                        except Exception as list_e:
                            logger.error(f"  Could not list contents: {list_e}")
            except Exception as debug_e:
                logger.error(f"Error during import debugging: {debug_e}")
            
            return False
    
    def _load_refresh_token_classes(self) -> Tuple[Any, Any]:
        """Load RefreshTokenBase and RefreshTokenService classes"""
        
        # Try direct import first
        try:
            from shared.refresh_token_service import RefreshTokenBase, RefreshTokenService
            logger.info("Successfully imported from shared.refresh_token_service")
            return RefreshTokenBase, RefreshTokenService
        except ImportError as e:
            logger.error(f"Direct import failed: {e}")
            
            # Try manual loading
            return self._manual_load_refresh_token_classes()
    
    def _manual_load_refresh_token_classes(self) -> Tuple[Any, Any]:
        """Manually load refresh token classes from file"""
        
        # Find shared directory
        shared_dir = None
        possible_paths = [
            '/app/shared',
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'),
            os.path.join(os.getcwd(), 'shared'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'refresh_token_service.py')):
                shared_dir = path
                break
        
        if not shared_dir:
            logger.error("Could not find shared directory with refresh_token_service.py")
            return None, None
        
        try:
            # Manual module loading
            refresh_file = os.path.join(shared_dir, 'refresh_token_service.py')
            spec = importlib.util.spec_from_file_location("refresh_token_service", refresh_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                RefreshTokenBase = getattr(module, 'RefreshTokenBase', None)
                RefreshTokenService = getattr(module, 'RefreshTokenService', None)
                
                if RefreshTokenBase and RefreshTokenService:
                    logger.info("Successfully loaded classes via manual import")
                    return RefreshTokenBase, RefreshTokenService
                else:
                    logger.error("Classes not found in module")
                    return None, None
            else:
                logger.error("Could not load refresh_token_service.py")
                return None, None
                
        except Exception as e:
            logger.error(f"Manual loading failed: {e}")
            return None, None
    
    def _create_fallback_implementation(self) -> Tuple[Any, Any]:
        """Create minimal fallback implementation"""
        logger.warning("Creating minimal fallback RefreshTokenService implementation")
        
        try:
            import secrets
            from sqlalchemy.ext.declarative import declarative_base
            
            class FallbackRefreshToken:
                """Minimal fallback refresh token record"""
                def __init__(self, customer_id, token_hash, device_id=None, user_agent=None, ip_address=None):
                    self.customer_id = customer_id
                    self.token_hash = token_hash
                    self.device_id = device_id
                    self.user_agent = user_agent
                    self.ip_address = ip_address
            
            class FallbackRefreshTokenService:
                def __init__(self, db, redis_client=None):
                    self.db = db
                    self.redis_client = redis_client
                    logger.warning("Using fallback RefreshTokenService implementation")
                
                def create_refresh_token(self, customer_id, device_id=None, user_agent=None, ip_address=None, **kwargs):
                    logger.warning("create_refresh_token called with fallback implementation")
                    # Generate a secure token for fallback
                    refresh_token = secrets.token_urlsafe(32)
                    token_record = FallbackRefreshToken(
                        customer_id=customer_id,
                        token_hash=refresh_token,  # In fallback, we use the token directly
                        device_id=device_id,
                        user_agent=user_agent,
                        ip_address=ip_address
                    )
                    return refresh_token, token_record
                
                def validate_token(self, *args, **kwargs):
                    logger.warning("validate_token called with fallback implementation")
                    return None
                
                def revoke_token(self, *args, **kwargs):
                    logger.warning("revoke_token called with fallback implementation")
                    return None
                
                def revoke_all_tokens(self, customer_id, reason="fallback"):
                    logger.warning("revoke_all_tokens called with fallback implementation")
                    return 0
                
                def rotate_token(self, *args, **kwargs):
                    logger.warning("rotate_token called with fallback implementation")
                    return None
            
            RefreshTokenBase = declarative_base()
            return RefreshTokenBase, FallbackRefreshTokenService
            
        except Exception as e:
            logger.error(f"Failed to create fallback implementation: {e}")
            return None, None
    
    def initialize(self) -> Tuple[Any, Any]:
        """Initialize shared module imports and return classes"""
        
        logger.info("Initializing shared module imports...")
        
        # Try to import shared module
        if not self._import_shared_module():
            logger.error("Could not import shared module")
            return self._create_fallback_implementation()
        
        # Load refresh token classes
        RefreshTokenBase, RefreshTokenService = self._load_refresh_token_classes()
        
        if RefreshTokenBase and RefreshTokenService:
            logger.info("Successfully loaded all required classes")
            self.shared_loaded = True
            self.refresh_token_base = RefreshTokenBase
            self.refresh_token_service = RefreshTokenService
            return RefreshTokenBase, RefreshTokenService
        else:
            logger.error("Failed to load required classes, using fallback")
            return self._create_fallback_implementation()
    
    def is_shared_available(self) -> bool:
        """Check if shared module is properly loaded"""
        return self.shared_loaded
    
    def get_refresh_token_service(self):
        """Get the RefreshTokenService class"""
        return self.refresh_token_service
    
    def get_refresh_token_base(self):
        """Get the RefreshTokenBase class"""
        return self.refresh_token_base

# Global instance
_importer = SharedModuleImporter()

def get_shared_classes():
    """Convenience function to get shared classes"""
    return _importer.initialize()

def is_shared_available():
    """Check if shared module is available"""
    return _importer.is_shared_available()