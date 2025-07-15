#!/bin/bash
set -e

# Railway-specific PORT handling
echo "=== Railway Deployment Debug ==="
echo "Raw PORT variable: '$PORT'"
echo "Environment check:"
env | grep -i port || echo "No PORT found in environment"

# Handle Railway's dynamic PORT assignment
if [ -z "$PORT" ] || ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    # If PORT is empty or not a valid number, use default
    PORT=8000
    echo "⚠️  PORT not properly set or invalid, using default: $PORT"
else
    echo "✅ Using Railway PORT: $PORT"
fi

echo "🚀 Starting Customer Service on port $PORT"
echo "📡 Uvicorn command: uvicorn src.main:app --host 0.0.0.0 --port $PORT"

# Start the FastAPI application
exec uvicorn src.main:app --host 0.0.0.0 --port "$PORT"
