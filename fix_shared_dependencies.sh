#!/bin/bash

# Script to fix Dockerfiles to include shared dependencies
set -e

echo "🔧 Fixing Dockerfiles to include shared dependencies..."

# List of services to update (excluding api-gateway which has special config)
SERVICES=(
    "auth-service"
    "customer-service"
    "reminder-service"
    "note-service"
    "ledger-service"
    "friend-service"
    "history-service"
    "chat-service"
    "scheduler-service"
    "stt-service"
    "tts-service"
    "intent-service"
    "ai-pipeline-service"
)

for service in "${SERVICES[@]}"; do
    echo "📝 Updating $service Dockerfile..."
    
    # Create standardized Dockerfile content with shared dependencies
    cat > "services/$service/Dockerfile" << 'EOF'
FROM python:3.11-slim

# Add build argument to force rebuild
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# Set default port
ENV PORT=8000

# Ensure the working directory is set to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and shared dependencies
COPY src/ ./src/
COPY ../../shared/ ./shared/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE $PORT

# Copy and make executable
COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run with environment variables
CMD ["/app/start.sh"]
EOF

    echo "✅ Updated $service"
done

echo "🎉 All Dockerfiles updated with shared dependencies!"
echo ""
echo "📋 Key fixes applied:"
echo "1. ✅ Added COPY src/ ./src/"
echo "2. ✅ Added COPY ../../shared/ ./shared/"
echo "3. ✅ Added COPY migrations/ ./migrations/"
echo "4. ✅ Added COPY alembic.ini ."
echo "5. ✅ Set PYTHONPATH=/app/src"
echo "6. ✅ Maintained start.sh functionality"
echo ""
echo "🔧 Next steps:"
echo "1. Rebuild all services:"
echo "   docker-compose -f docker-compose.microservices.yml build"
echo ""
echo "2. Start the services:"
echo "   docker-compose -f docker-compose.microservices.yml up -d" 