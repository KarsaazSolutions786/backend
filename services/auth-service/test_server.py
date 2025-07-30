#!/usr/bin/env python3

import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

if __name__ == "__main__":
    print("Starting test server...")
    try:
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8002,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"Server failed to start: {e}")
        import traceback
        traceback.print_exc()