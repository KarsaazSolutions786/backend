#!/bin/bash

# Script to fix all Dockerfiles with standardized configuration
set -e

echo "🔧 Fixing all Dockerfiles..."

# List of services to update (excluding api-gateway which has special config)
SERVICES=(
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
    
    # Create standardized Dockerfile content
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

# Copy the application code
COPY . .

# Install dependencies and set up the environment
RUN pip install -r requirements.txt

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

echo "🎉 All Dockerfiles updated successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Make all start.sh files executable:"
echo "   find services/ -name 'start.sh' -exec chmod +x {} \;"
echo ""
echo "2. Rebuild all services:"
echo "   docker-compose -f docker-compose.microservices.yml build"
echo ""
echo "3. Start the services:"
echo "   docker-compose -f docker-compose.microservices.yml up -d" 