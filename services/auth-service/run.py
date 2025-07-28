#!/usr/bin/env python3
"""
Simple runner script for the auth service
"""

import os
import sys
import uvicorn

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add the backend root directory to Python path for shared modules
backend_root = os.path.join(current_dir, '../..')
sys.path.insert(0, os.path.abspath(backend_root))

# Import the FastAPI app
from src.main import app

if __name__ == "__main__":
    # Get port from environment variable (Railway sets PORT)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Auth Service on {host}:{port}")
    
    # Run the application
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False  # Disable reload in production
    )