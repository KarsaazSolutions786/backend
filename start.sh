#!/bin/bash
# Railway startup script
# This script ensures the PORT environment variable is properly handled

# Set default port if not provided
export PORT=${PORT:-8000}

echo "Starting Auth Service on port $PORT"

# Start the FastAPI application (using correct path from project structure)
exec uvicorn src.main:app --host 0.0.0.0 --port $PORT 