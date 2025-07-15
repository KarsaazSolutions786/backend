#!/usr/bin/env python3
"""
Railway-compatible startup script for Customer Service
Handles PORT environment variable properly for Railway deployment
"""

import os
import sys
import uvicorn

def main():
    # Debug: Print environment information
    print("=== Railway Deployment Debug ===")
    print(f"Raw PORT variable: '{os.getenv('PORT', 'NOT_SET')}'")
    print(f"All environment variables containing 'PORT':")
    for key, value in os.environ.items():
        if 'PORT' in key.upper():
            print(f"  {key}={value}")
    
    # Handle PORT environment variable
    try:
        port_str = os.getenv('PORT')
        if port_str is None:
            port = 8000
            print("⚠️  PORT not set, using default: 8000")
        else:
            port = int(port_str)
            print(f"✅ Using Railway PORT: {port}")
    except (ValueError, TypeError) as e:
        print(f"❌ Invalid PORT value: '{port_str}' - {e}")
        print("🔄 Falling back to default port: 8000")
        port = 8000
    
    # Validate port range
    if not (1 <= port <= 65535):
        print(f"❌ PORT {port} is out of valid range (1-65535)")
        print("🔄 Falling back to default port: 8000")
        port = 8000
    
    print(f"🚀 Starting Customer Service on port {port}")
    print(f"📡 Uvicorn command: uvicorn src.main:app --host 0.0.0.0 --port {port}")
    
    # Start the FastAPI application
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 