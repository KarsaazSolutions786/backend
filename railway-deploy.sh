#!/bin/bash

# Railway Deployment Script for Eindr Microservices
# This script helps deploy your microservices to Railway

set -e

echo "🚀 Starting Railway Deployment for Eindr Microservices"

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

echo "📋 Available commands:"
echo "1. Deploy main API gateway"
echo "2. Deploy individual services"
echo "3. Set up database"
echo "4. Configure environment variables"
echo "5. View deployment status"

read -p "Choose an option (1-5): " choice

case $choice in
    1)
        echo "🚀 Deploying main API gateway..."
        railway up
        ;;
    2)
        echo "🔧 Deploying individual services..."
        echo "Note: You'll need to create separate Railway projects for each service"
        echo "Recommended services to deploy:"
        echo "- auth-service"
        echo "- chat-service" 
        echo "- customer-service"
        echo "- reminder-service"
        ;;
    3)
        echo "🗄️ Setting up database..."
        echo "1. Go to your Railway project dashboard"
        echo "2. Click 'New Service'"
        echo "3. Select 'Database' → 'PostgreSQL'"
        echo "4. Copy the DATABASE_URL to your environment variables"
        ;;
    4)
        echo "⚙️ Configuring environment variables..."
        echo "Required environment variables:"
        echo "- DATABASE_URL"
        echo "- SECRET_KEY"
        echo "- JWT_SECRET"
        echo "- ALLOWED_ORIGINS"
        echo "- ENVIRONMENT=production"
        ;;
    5)
        echo "📊 Deployment status:"
        railway status
        railway logs
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo "✅ Deployment process completed!"
echo "🌐 Your app should be available at: https://your-app-name.railway.app" 