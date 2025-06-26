#!/bin/bash

# Kong Admin API URL
KONG_ADMIN_URL="http://localhost:8101"

# Wait for Kong to be ready
echo "⏳ Waiting for Kong to be ready..."
until curl -f $KONG_ADMIN_URL/status > /dev/null 2>&1; do
    echo "Kong not ready yet, waiting..."
    sleep 5
done
echo "✅ Kong is ready!"

# Function to create service
create_service() {
    local name=$1
    local url=$2
    local path=$3
    
    echo "🔧 Creating service: $name"
    curl -i -X POST $KONG_ADMIN_URL/services/ \
        --data "name=$name" \
        --data "url=$url" \
        --data "retries=3" \
        --data "connect_timeout=60000" \
        --data "write_timeout=60000" \
        --data "read_timeout=60000"
    
    echo ""
    echo "🛣️  Creating route for service: $name"
    curl -i -X POST $KONG_ADMIN_URL/services/$name/routes \
        --data "paths[]=$path" \
        --data "strip_path=false" \
        --data "preserve_host=false"
    echo ""
}

# Function to add JWT plugin to service
add_jwt_plugin() {
    local service_name=$1
    
    echo "🔐 Adding JWT plugin to service: $service_name"
    curl -i -X POST $KONG_ADMIN_URL/services/$service_name/plugins \
        --data "name=jwt" \
        --data "config.secret_is_base64=false" \
        --data "config.key_claim_name=iss" \
        --data "config.claims_to_verify=exp"
    echo ""
}

# Function to add CORS plugin to service
add_cors_plugin() {
    local service_name=$1
    
    echo "🌐 Adding CORS plugin to service: $service_name"
    curl -i -X POST $KONG_ADMIN_URL/services/$service_name/plugins \
        --data "name=cors" \
        --data "config.origins=*" \
        --data "config.methods=GET,POST,PUT,DELETE,OPTIONS,PATCH" \
        --data "config.headers=Accept,Accept-Version,Content-Length,Content-MD5,Content-Type,Date,Authorization" \
        --data "config.exposed_headers=X-Auth-Token" \
        --data "config.credentials=true" \
        --data "config.max_age=3600"
    echo ""
}

# Function to add rate limiting plugin
add_rate_limiting() {
    local service_name=$1
    local minute_limit=${2:-100}
    local hour_limit=${3:-1000}
    
    echo "⏱️  Adding rate limiting to service: $service_name"
    curl -i -X POST $KONG_ADMIN_URL/services/$service_name/plugins \
        --data "name=rate-limiting" \
        --data "config.minute=$minute_limit" \
        --data "config.hour=$hour_limit" \
        --data "config.policy=local"
    echo ""
}

# Create services and routes
echo "🚀 Setting up Kong services and routes..."

# Auth Service (no JWT required for this service as it provides authentication)
create_service "auth-service" "http://auth-service:8000" "/auth"
add_cors_plugin "auth-service"
add_rate_limiting "auth-service" 60 600

# User Service
create_service "user-service" "http://user-service:8000" "/users"
add_jwt_plugin "user-service"
add_cors_plugin "user-service"
add_rate_limiting "user-service" 100 1000

# Reminder Service
create_service "reminder-service" "http://reminder-service:8000" "/reminders"
add_jwt_plugin "reminder-service"
add_cors_plugin "reminder-service"
add_rate_limiting "reminder-service" 150 1500

# Note Service
create_service "note-service" "http://note-service:8000" "/notes"
add_jwt_plugin "note-service"
add_cors_plugin "note-service"
add_rate_limiting "note-service" 200 2000

# Ledger Service
create_service "ledger-service" "http://ledger-service:8000" "/expenses"
add_jwt_plugin "ledger-service"
add_cors_plugin "ledger-service"
add_rate_limiting "ledger-service" 100 1000

# Friend Service
create_service "friend-service" "http://friend-service:8000" "/friends"
add_jwt_plugin "friend-service"
add_cors_plugin "friend-service"
add_rate_limiting "friend-service" 50 500

# History Service
create_service "history-service" "http://history-service:8000" "/logs"
add_jwt_plugin "history-service"
add_cors_plugin "history-service"
add_rate_limiting "history-service" 50 500

# STT Service
create_service "stt-service" "http://stt-service:8000" "/stt"
add_jwt_plugin "stt-service"
add_cors_plugin "stt-service"
add_rate_limiting "stt-service" 30 300  # Lower limit for resource-intensive operations

# TTS Service
create_service "tts-service" "http://tts-service:8000" "/tts"
add_jwt_plugin "tts-service"
add_cors_plugin "tts-service"
add_rate_limiting "tts-service" 30 300  # Lower limit for resource-intensive operations

# Intent Service
create_service "intent-service" "http://intent-service:8000" "/intent"
add_jwt_plugin "intent-service"
add_cors_plugin "intent-service"
add_rate_limiting "intent-service" 100 1000

# Chat Service
create_service "chat-service" "http://chat-service:8000" "/conversations"
add_jwt_plugin "chat-service"
add_cors_plugin "chat-service"
add_rate_limiting "chat-service" 200 2000

# Scheduler Service
create_service "scheduler-service" "http://scheduler-service:8000" "/jobs"
add_jwt_plugin "scheduler-service"
add_cors_plugin "scheduler-service"
add_rate_limiting "scheduler-service" 50 500

# Create JWT consumer for auth service
echo "🔑 Creating JWT consumer for authentication..."
curl -i -X POST $KONG_ADMIN_URL/consumers/ \
    --data "username=eindr-auth-service"

curl -i -X POST $KONG_ADMIN_URL/consumers/eindr-auth-service/jwt \
    --data "key=eindr-issuer" \
    --data "secret=your-jwt-secret-key-change-in-production"

# Health check route (no authentication required)
echo "🏥 Creating health check route..."
create_service "health-service" "http://auth-service:8000" "/health"
add_cors_plugin "health-service"

# Add global plugins
echo "🌍 Adding global plugins..."

# Global request ID plugin
curl -i -X POST $KONG_ADMIN_URL/plugins/ \
    --data "name=correlation-id" \
    --data "config.header_name=X-Request-ID" \
    --data "config.generator=uuid"

# Global logging plugin
curl -i -X POST $KONG_ADMIN_URL/plugins/ \
    --data "name=file-log" \
    --data "config.path=/tmp/access.log"

echo ""
echo "🎉 Kong setup completed!"
echo ""
echo "📋 Available endpoints:"
echo "🌐 API Gateway: http://localhost:8080"
echo "⚙️  Kong Admin API: http://localhost:8001"
echo "📊 Kong Admin GUI: http://localhost:8102"
echo "🖥️  Konga UI: http://localhost:8103"
echo ""
echo "📚 Service endpoints via gateway:"
echo "🔐 Auth: http://localhost:8080/auth/*"
echo "👤 Users: http://localhost:8080/users/*"
echo "⏰ Reminders: http://localhost:8080/reminders/*"
echo "📝 Notes: http://localhost:8080/notes/*"
echo "💰 Expenses: http://localhost:8080/expenses/*"
echo "👥 Friends: http://localhost:8080/friends/*"
echo "📊 Logs: http://localhost:8080/logs/*"
echo "🎤 STT: http://localhost:8080/stt/*"
echo "🔊 TTS: http://localhost:8080/tts/*"
echo "🧠 Intent: http://localhost:8080/intent/*"
echo "💬 Chat: http://localhost:8080/conversations/*"
echo "⚡ Jobs: http://localhost:8080/jobs/*"
echo "🏥 Health: http://localhost:8080/health" 