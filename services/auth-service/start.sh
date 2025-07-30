#!/bin/bash
set -e

# Set correct PYTHONPATH for Railway deployment
export PYTHONPATH="/app:/app/src:/app/shared"

# Debug: Print environment variables
echo "Debug: Environment variables:"
echo "PORT=$PORT"
echo "PYTHONPATH=$PYTHONPATH"

# Debug: Check shared module availability
echo "Debug: Checking shared module structure:"
echo "Contents of /app:"
ls -la /app/ || echo "Failed to list /app"
echo "Contents of /app/shared:"
ls -la /app/shared/ || echo "Failed to list /app/shared"
echo "Testing shared module import:"
python -c "import sys; print('Python sys.path:', sys.path)" || echo "Failed to print sys.path"
python -c "import shared; print('Shared module imported successfully')" || echo "Failed to import shared module"
echo "Testing refresh_token_service import:"
python -c "from shared.refresh_token_service import RefreshTokenBase, RefreshTokenService; print('RefreshTokenService imported successfully')" || echo "Failed to import RefreshTokenService"

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
