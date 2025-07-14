#!/usr/bin/env python3
"""
Simple runner script for the customer service
"""

import os
import sys
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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