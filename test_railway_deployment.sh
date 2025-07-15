#!/bin/bash

# Test script to simulate Railway deployment locally
echo "🚀 Testing Railway deployment configuration..."

# Test 1: Build the Docker image
echo "📦 Building Docker image..."
docker build -f services/customer-service/Dockerfile -t customer-service-test .

# Test 2: Run with Railway-like environment
echo "🔧 Running with Railway-like environment..."
docker run --rm \
  -e PORT=12345 \
  -e DATABASE_URL=postgresql://test:test@localhost:5432/test \
  -e JWT_SECRET=test-secret \
  -e ENVIRONMENT=production \
  -p 12345:12345 \
  customer-service-test &

# Wait for container to start
sleep 5

# Test 3: Check if service is running
echo "🔍 Checking if service is running..."
curl -f http://localhost:12345/health || echo "❌ Service not responding"

# Test 4: Check logs
echo "📋 Container logs:"
docker logs $(docker ps -q --filter ancestor=customer-service-test) 2>/dev/null || echo "No logs found"

# Cleanup
echo "🧹 Cleaning up..."
docker stop $(docker ps -q --filter ancestor=customer-service-test) 2>/dev/null || true
docker rmi customer-service-test 2>/dev/null || true

echo "✅ Test complete!" 