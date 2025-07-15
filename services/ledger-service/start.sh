#!/bin/bash
set -e

# Default to port 8000 if $PORT is not set
PORT=${PORT:-8000}

# Ensure PORT is an integer
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: PORT must be a number. Got '$PORT'"
  exit 1
fi

echo "Starting Ledger Service on port $PORT"

# Start the FastAPI application
exec uvicorn src.main:app --host 0.0.0.0 --port $PORT 