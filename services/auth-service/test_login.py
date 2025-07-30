#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    print("Testing imports...")
    from src.main import app
    print("✓ Main app imported successfully")
    
    from src.routers.auth import router
    print("✓ Auth router imported successfully")
    
    from src.services.auth_business_service import AuthBusinessService
    print("✓ AuthBusinessService imported successfully")
    
    from src.database import get_db
    print("✓ Database connection imported successfully")
    
    print("\nAll imports successful! Testing database connection...")
    
    # Test database connection
    db_gen = get_db()
    db = next(db_gen)
    print("✓ Database connection established")
    
    # Test AuthBusinessService initialization
    auth_service = AuthBusinessService(db)
    print("✓ AuthBusinessService initialized successfully")
    
    print("\n🎉 All tests passed! The service should work correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)