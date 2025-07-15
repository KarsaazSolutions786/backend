#!/bin/bash
set -e

echo "Raw PORT variable: '$PORT'"
# If PORT is not set or not a number, default to 8000
if [[ -z "$PORT" || ! "$PORT" =~ ^[0-9]+$ ]]; then
  PORT=8000
  echo "⚠️  PORT not set or invalid, defaulting to $PORT"
else
  echo "✅ Using PORT $PORT"
fi

echo "Starting Uvicorn on port $PORT..."
exec uvicorn src.main:app --host 0.0.0.0 --port "$PORT"
