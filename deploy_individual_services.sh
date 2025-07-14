#!/bin/bash

# Individual Microservice Deployment Script for Railway
# This script helps deploy each microservice separately to Railway

set -e

echo "🚀 Starting Individual Microservice Deployment to Railway"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway first:"
    railway login
fi

# Service configurations
declare -A services=(
    ["auth-service"]="eindr-auth"
    ["customer-service"]="eindr-customers"
    ["reminder-service"]="eindr-reminders"
    ["note-service"]="eindr-notes"
    ["ledger-service"]="eindr-ledger"
    ["friend-service"]="eindr-friends"
    ["history-service"]="eindr-history"
    ["stt-service"]="eindr-stt"
    ["tts-service"]="eindr-tts"
    ["intent-service"]="eindr-intent"
    ["chat-service"]="eindr-chat"
    ["scheduler-service"]="eindr-scheduler"
)

echo "📋 Available services to deploy:"
echo "1. Deploy all services"
echo "2. Deploy specific service"
echo "3. Deploy API Gateway"
echo "4. List service URLs"
echo "5. Check deployment status"

read -p "Choose an option (1-5): " choice

case $choice in
    1)
        echo "🚀 Deploying all services..."
        for service in "${!services[@]}"; do
            echo "📦 Deploying $service..."
            cd "services/$service"
            railway init --name "${services[$service]}" --yes
            railway up
            cd ../..
            echo "✅ $service deployed successfully!"
        done
        echo "🎉 All services deployed!"
        ;;
    2)
        echo "🔧 Available services:"
        i=1
        for service in "${!services[@]}"; do
            echo "$i. $service"
            ((i++))
        done
        read -p "Choose service number: " service_num
        i=1
        for service in "${!services[@]}"; do
            if [ $i -eq $service_num ]; then
                echo "📦 Deploying $service..."
                cd "services/$service"
                railway init --name "${services[$service]}" --yes
                railway up
                cd ../..
                echo "✅ $service deployed successfully!"
                break
            fi
            ((i++))
        done
        ;;
    3)
        echo "🌐 Deploying API Gateway..."
        railway init --name "eindr-gateway" --yes
        railway up
        echo "✅ API Gateway deployed successfully!"
        ;;
    4)
        echo "🔗 Service URLs (after deployment):"
        for service in "${!services[@]}"; do
            echo "$service: https://${services[$service]}.railway.app"
        done
        echo "API Gateway: https://eindr-gateway.railway.app"
        ;;
    5)
        echo "📊 Checking deployment status..."
        railway status
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo "✅ Deployment process completed!"
echo ""
echo "📝 Next steps:"
echo "1. Configure environment variables in Railway dashboard"
echo "2. Add PostgreSQL databases for services that need them"
echo "3. Run database migrations: railway shell && alembic upgrade head"
echo "4. Test your services at their individual URLs"
echo "5. Update API Gateway with service URLs" 