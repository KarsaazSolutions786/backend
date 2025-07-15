#!/bin/bash
set -e

# Railway-specific PORT handling
echo "=== Railway Deployment Debug ==="
echo "Raw PORT variable: '$PORT'"
echo "Environment check:"
env | grep -i port || echo "No PORT found in environment"

# Handle Railway's dynamic PORT assignment
if [ -z "$PORT" ] || [ "$PORT" = "$PORT" ]; then
    # If PORT is empty or literally "$PORT", use default
    PORT=8000
    echo "⚠️  PORT not properly set, using default: $PORT"
else
    echo "✅ Using Railway PORT: $PORT"
fi

# Ensure PORT is a valid integer
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ Invalid PORT value: '$PORT'"
    echo "🔄 Falling back to default port: 8000"
    PORT=8000
fi

echo "🚀 Starting Customer Service on port $PORT"
echo "📡 Uvicorn command: uvicorn src.main:app --host 0.0.0.0 --port $PORT"

# Start the FastAPI application
exec uvicorn src.main:app --host 0.0.0.0 --port "$PORT"
