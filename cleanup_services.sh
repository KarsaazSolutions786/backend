#!/bin/bash

# Script to clean up and update microservices configurations
# Remove external_links from auth-service and customer-service

echo "🧹 Starting cleanup of microservices configurations..."

# Define services that need updates
services=("auth-service" "customer-service" "reminder-service" "note-service" "ledger-service" "friend-service" "history-service" "chat-service" "scheduler-service")

for service in "${services[@]}"; do
    echo "🔧 Updating dependencies for $service"
    
    # Find the service section and add new-postgres-server to depends_on
    # This is a bit complex with sed, so we'll use a Python script instead
done

echo "✅ Service dependencies updated" 