#!/usr/bin/env python3
"""
Production server startup script for Eindr Backend
Handles both Railway deployment and local development
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Main startup function with Railway optimization."""
    
    # Configuration
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Check if we're in Railway
    is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
    is_minimal = os.getenv("MINIMAL_MODE", "false").lower() == "true"
    
    # Force minimal mode for Railway
    if is_railway:
        is_minimal = True
        os.environ["MINIMAL_MODE"] = "true"
        print("🚀 Railway deployment detected - using minimal mode")
    
    print(f"🚀 Starting Eindr Backend...")
    print(f"Environment: {'Railway' if is_railway else 'Local'}")
    print(f"Mode: {'Minimal' if is_minimal else 'Full'}")
    print(f"Port: {port}")
    
    # Railway-optimized configuration
    if is_railway:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=1,
            timeout_keep_alive=30,
            server_header=False,
            proxy_headers=True,
            forwarded_allow_ips="*"
        )
    else:
        # Local/full mode configuration
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=False,
            workers=1,
            timeout_keep_alive=30,
            limit_concurrency=100,
            limit_max_requests=1000,
            server_header=False,
            proxy_headers=True
        )

if __name__ == "__main__":
    main() 