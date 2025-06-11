#!/usr/bin/env python3
"""
Startup script for Eindr backend with proper environment setup
"""
import os
import sys

# CRITICAL: Set environment variables BEFORE importing any modules
os.environ["MINIMAL_MODE"] = "false"
print(f"🤖 MINIMAL_MODE set to: {os.environ.get('MINIMAL_MODE')}")

# Verify Python version and environment
print(f"🐍 Python version: {sys.version}")
print(f"📁 Python executable: {sys.executable}")

# Check for PyTorch availability
try:
    import torch
    print(f"✅ PyTorch available: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not available!")
    sys.exit(1)

try:
    import transformers
    print(f"✅ Transformers available: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not available!")
    sys.exit(1)

# Now import the main application
print("🚀 Starting Eindr backend with Bloom-560M...")

if __name__ == "__main__":
    import uvicorn
    from main import app
    from core.config import settings
    
    print(f"🔧 Settings MINIMAL_MODE: {settings.MINIMAL_MODE}")
    print(f"🔧 Chat model path: {settings.CHAT_MODEL_PATH}")
    
    # Start the server
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # Disable reload to prevent environment reset
        log_level="info"
    ) 