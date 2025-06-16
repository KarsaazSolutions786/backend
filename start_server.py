#!/usr/bin/env python3
"""
Production server startup script for Eindr Backend
Handles both Railway deployment and local development
"""

import os
import sys
import subprocess
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
    print(f"Python executable: {sys.executable}")
    
    # Try different uvicorn approaches
    uvicorn_methods = [
        # Method 1: Direct uvicorn import (preferred)
        lambda: _start_with_uvicorn_import(host, port, is_railway),
        # Method 2: uvicorn via subprocess with full path
        lambda: _start_with_subprocess_fullpath(host, port, is_railway),
        # Method 3: uvicorn via python -m
        lambda: _start_with_python_module(host, port, is_railway)
    ]
    
    for i, method in enumerate(uvicorn_methods, 1):
        try:
            print(f"📡 Attempting startup method {i}...")
            method()
            break  # If we get here, it worked
        except Exception as e:
            print(f"❌ Startup method {i} failed: {e}")
            if i == len(uvicorn_methods):
                print("💥 All startup methods failed!")
                sys.exit(1)
            print(f"🔄 Trying next method...")

def _start_with_uvicorn_import(host, port, is_railway):
    """Start using uvicorn import (most reliable)."""
    import uvicorn
    
    config = {
        "app": "main:app",
        "host": host,
        "port": port,
        "workers": 1,
        "timeout_keep_alive": 30,
        "server_header": False,
        "access_log": True
    }
    
    if is_railway:
        config.update({
            "proxy_headers": True,
            "forwarded_allow_ips": "*"
        })
    else:
        config.update({
            "limit_concurrency": 100,
            "limit_max_requests": 1000,
            "proxy_headers": True
        })
    
    print("✅ Starting with uvicorn.run()...")
    uvicorn.run(**config)

def _start_with_subprocess_fullpath(host, port, is_railway):
    """Start using subprocess with full uvicorn path."""
    uvicorn_path = "/opt/venv/bin/uvicorn"
    if not os.path.exists(uvicorn_path):
        raise FileNotFoundError(f"uvicorn not found at {uvicorn_path}")
    
    cmd = [
        uvicorn_path, "main:app",
        "--host", host,
        "--port", str(port),
        "--workers", "1",
        "--timeout-keep-alive", "30"
    ]
    
    if is_railway:
        cmd.extend(["--proxy-headers", "--forwarded-allow-ips", "*"])
    
    print(f"✅ Starting with subprocess: {' '.join(cmd)}")
    os.execv(uvicorn_path, cmd)

def _start_with_python_module(host, port, is_railway):
    """Start using python -m uvicorn."""
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", host,
        "--port", str(port),
        "--workers", "1",
        "--timeout-keep-alive", "30"
    ]
    
    if is_railway:
        cmd.extend(["--proxy-headers", "--forwarded-allow-ips", "*"])
    
    print(f"✅ Starting with python -m: {' '.join(cmd)}")
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main() 