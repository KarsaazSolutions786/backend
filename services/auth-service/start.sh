#!/bin/bash
set -e

# Set correct PYTHONPATH for Railway deployment
export PYTHONPATH="/app:/app/src:/app/shared"

# Debug: Print environment variables
echo "Debug: Environment variables:"
echo "PORT=$PORT"
echo "PYTHONPATH=$PYTHONPATH"

# Create shared directory if it doesn't exist (fallback)
if [ ! -d "/app/shared" ]; then
    echo "Creating shared directory..."
    mkdir -p /app/shared
fi

# List contents to verify shared module
echo "=== Directory Structure Check ==="
echo "Contents of /app:"
ls -la /app/
echo "Contents of /app/shared:"
ls -la /app/shared/ || echo "Shared directory empty or not found"

# Run diagnostic script to test shared module imports
echo "Running import diagnostics..."
python test_import.py

echo "Current working directory: $(pwd)"
echo "Backend path: $(dirname $(pwd))"
echo "Python path: $(python -c "import sys; print(sys.path)")"

# Default to port 8000 if $PORT is not set or empty
if [ -z "$PORT" ]; then
    PORT=8000
    echo "PORT not set, using default: $PORT"
else
    echo "Using PORT from environment: $PORT"
fi

# Ensure PORT is an integer
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PORT must be a number. Got '$PORT'"
    echo "Setting PORT to default value: 8000"
    PORT=8000
fi

echo "Starting service on port $PORT"

# Start the FastAPI application with explicit port
exec uvicorn src.main:app --host 0.0.0.0 --port "$PORT"
