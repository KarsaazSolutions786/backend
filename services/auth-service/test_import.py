#!/usr/bin/env python3
"""
Comprehensive diagnostic script for shared module imports in Railway deployment
"""

import sys
import os
import importlib.util
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_diagnostics():
    """Run comprehensive diagnostics for shared module imports"""
    
    print("=" * 60)
    print("SHARED MODULE IMPORT DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Environment Information
    print("\n1. ENVIRONMENT INFORMATION")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
    
    # 2. Directory Structure
    print("\n2. DIRECTORY STRUCTURE")
    print("-" * 30)
    print("Contents of /app:")
    try:
        contents = os.listdir('/app')
        for item in sorted(contents):
            item_path = os.path.join('/app', item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except FileNotFoundError:
        print("  ❌ /app directory not found")
    
    # Check shared directory specifically
    print("\nContents of /app/shared:")
    try:
        shared_contents = os.listdir('/app/shared')
        for item in sorted(shared_contents):
            item_path = os.path.join('/app/shared', item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except FileNotFoundError:
        print("  ❌ /app/shared directory not found")
    
    # 3. Python Path Analysis
    print("\n3. PYTHON PATH ANALYSIS")
    print("-" * 30)
    for i, path in enumerate(sys.path):
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {i}: {exists} {path}")
    
    # 4. Shared Module Verification
    print("\n4. SHARED MODULE VERIFICATION")
    print("-" * 30)
    
    # Check if shared is importable
    try:
        import shared
        print("  ✅ shared package imported successfully")
        print(f"  📍 shared.__file__: {getattr(shared, '__file__', 'N/A')}")
        print(f"  📍 shared.__path__: {getattr(shared, '__path__', 'N/A')}")
    except ImportError as e:
        print(f"  ❌ Failed to import shared: {e}")
        
        # Try to find shared directory manually
        possible_paths = [
            '/app/shared',
            './shared',
            '../shared',
            '../../shared',
            os.path.join(os.getcwd(), 'shared')
        ]
        
        print("\n  Searching for shared directory...")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✅ Found shared at: {path}")
                print(f"  📂 Contents: {os.listdir(path)}")
            else:
                print(f"  ❌ Not found at: {path}")
    
    # 5. Refresh Token Service Verification
    print("\n5. REFRESH TOKEN SERVICE VERIFICATION")
    print("-" * 30)
    
    # Check refresh_token_service.py exists
    refresh_service_path = '/app/shared/refresh_token_service.py'
    if os.path.exists(refresh_service_path):
        print(f"  ✅ refresh_token_service.py exists at: {refresh_service_path}")
        print(f"  📏 File size: {os.path.getsize(refresh_service_path)} bytes")
    else:
        print(f"  ❌ refresh_token_service.py not found at: {refresh_service_path}")
    
    # Check __init__.py
    init_path = '/app/shared/__init__.py'
    if os.path.exists(init_path):
        print(f"  ✅ __init__.py exists at: {init_path}")
    else:
        print(f"  ❌ __init__.py not found at: {init_path}")
    
    # 6. Import Test
    print("\n6. IMPORT TEST")
    print("-" * 30)
    
    # Test importing specific classes
    try:
        from shared.refresh_token_service import RefreshTokenBase, RefreshTokenService
        print("  ✅ Successfully imported RefreshTokenBase and RefreshTokenService")
        print(f"  🎯 RefreshTokenBase: {RefreshTokenBase}")
        print(f"  🎯 RefreshTokenService: {RefreshTokenService}")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import RefreshTokenBase/RefreshTokenService: {e}")
        
        # Try manual loading
        print("\n  Attempting manual module loading...")
        try:
            spec = importlib.util.spec_from_file_location(
                "refresh_token_service", 
                "/app/shared/refresh_token_service.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                RefreshTokenBase = getattr(module, 'RefreshTokenBase', None)
                RefreshTokenService = getattr(module, 'RefreshTokenService', None)
                
                if RefreshTokenBase and RefreshTokenService:
                    print("  ✅ Manual loading successful")
                    return True
                else:
                    print("  ❌ Classes not found in manually loaded module")
                    return False
            else:
                print("  ❌ Could not load module specification")
                return False
        except Exception as e2:
            print(f"  ❌ Manual loading failed: {e2}")
            return False
    
    # 7. Permissions Check
    print("\n7. PERMISSIONS CHECK")
    print("-" * 30)
    
    try:
        shared_stat = os.stat('/app/shared')
        print(f"  📋 Shared directory permissions: {oct(shared_stat.st_mode)[-3:]}")
        
        refresh_stat = os.stat('/app/shared/refresh_token_service.py')
        print(f"  📋 refresh_token_service.py permissions: {oct(refresh_stat.st_mode)[-3:]}")
    except Exception as e:
        print(f"  ❌ Could not check permissions: {e}")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 60)
    
    success = False
    try:
        from shared.refresh_token_service import RefreshTokenBase, RefreshTokenService
        print("✅ ALL TESTS PASSED - Shared module imports working correctly")
        success = True
    except Exception as e:
        print(f"❌ TESTS FAILED - {e}")
        print("\nRECOMMENDATIONS:")
        print("1. Ensure shared/ directory exists in /app/shared")
        print("2. Ensure shared/__init__.py and shared/refresh_token_service.py exist")
        print("3. Check file permissions (should be readable)")
        print("4. Verify PYTHONPATH includes /app/shared")
        print("5. Check Docker COPY commands in Dockerfile")
    
    return success

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)