#!/usr/bin/env python3
"""
Test script to verify shared module imports work correctly
This can be run in the deployment environment to debug import issues
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_environment():
    """Test the current environment setup"""
    logger.info("=== Environment Test ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH environment variable: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"Python sys.path: {sys.path}")
    
    # Check if /app/shared exists
    shared_paths = ['/app/shared', './shared', '../shared', '../../shared']
    for path in shared_paths:
        if os.path.exists(path):
            logger.info(f"Found shared directory at: {path}")
            try:
                files = os.listdir(path)
                logger.info(f"  Contents: {files}")
                
                # Check for specific files
                init_file = os.path.join(path, '__init__.py')
                refresh_file = os.path.join(path, 'refresh_token_service.py')
                logger.info(f"  __init__.py exists: {os.path.exists(init_file)}")
                logger.info(f"  refresh_token_service.py exists: {os.path.exists(refresh_file)}")
            except Exception as e:
                logger.error(f"  Error listing contents: {e}")
        else:
            logger.info(f"Shared directory not found at: {path}")

def test_shared_import():
    """Test importing the shared module"""
    logger.info("=== Shared Module Import Test ===")
    
    try:
        import shared
        logger.info("✓ Successfully imported 'shared' module")
        logger.info(f"  Module file: {getattr(shared, '__file__', 'Unknown')}")
        logger.info(f"  Module version: {getattr(shared, '__version__', 'Unknown')}")
    except ImportError as e:
        logger.error(f"✗ Failed to import 'shared' module: {e}")
        return False
    
    try:
        from shared.refresh_token_service import RefreshTokenBase, RefreshTokenService
        logger.info("✓ Successfully imported RefreshTokenBase and RefreshTokenService")
        logger.info(f"  RefreshTokenBase: {RefreshTokenBase}")
        logger.info(f"  RefreshTokenService: {RefreshTokenService}")
    except ImportError as e:
        logger.error(f"✗ Failed to import from shared.refresh_token_service: {e}")
        return False
    
    return True

def test_shared_importer():
    """Test the custom shared importer utility"""
    logger.info("=== Shared Importer Test ===")
    
    try:
        from src.utils.shared_importer import get_shared_classes, is_shared_available
        logger.info("✓ Successfully imported shared_importer utilities")
        
        # Test getting shared classes
        RefreshTokenBase, RefreshTokenService = get_shared_classes()
        logger.info(f"  RefreshTokenBase from importer: {RefreshTokenBase}")
        logger.info(f"  RefreshTokenService from importer: {RefreshTokenService}")
        
        # Check if shared is available
        available = is_shared_available()
        logger.info(f"  Shared module available: {available}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error testing shared importer: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting shared module import tests...")
    
    test_environment()
    
    shared_import_success = test_shared_import()
    importer_success = test_shared_importer()
    
    logger.info("=== Test Summary ===")
    logger.info(f"Direct shared import: {'✓ PASS' if shared_import_success else '✗ FAIL'}")
    logger.info(f"Shared importer utility: {'✓ PASS' if importer_success else '✗ FAIL'}")
    
    if shared_import_success and importer_success:
        logger.info("🎉 All tests passed! Shared module is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)