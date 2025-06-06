#!/bin/bash

# Railway Deployment Script for Eindr Backend
# This script helps deploy your application to Railway

echo "🚀 Starting Railway Deployment for Eindr Backend"
echo "================================================="

# Step 1: Verify we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Are you in the correct directory?"
    exit 1
fi

# Step 2: Check if git repository is clean
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes."
    echo "📝 Committing current changes..."
    git add .
    git commit -m "Update dependencies and fix Railway deployment issues"
fi

# Step 3: Push to repository
echo "📤 Pushing to repository..."
git push

echo "✅ Code pushed successfully!"
echo ""
echo "🔧 Next Steps:"
echo "1. Go to https://railway.app"
echo "2. Click 'Deploy from GitHub repo'"
echo "3. Select your repository"
echo "4. Add PostgreSQL service"
echo "5. Set these environment variables:"
echo "   - SECRET_KEY=8LrIcmpF1_QFIfGlLY6KtpvftqC4Co4mK4KyPOwrtOE"
echo "   - DEBUG=false"
echo "   - DEV_MODE=false"
echo "   - MINIMAL_MODE=true"
echo "   - RAILWAY_ENVIRONMENT=production"
echo ""
echo "6. Railway will automatically:"
echo "   - Set PORT environment variable"
echo "   - Set DATABASE_URL when PostgreSQL is added"
echo ""
echo "🎉 Your app will be available at: https://your-app.railway.app"
echo "📚 API docs: https://your-app.railway.app/docs"
echo ""
echo "🔍 Monitor deployment: Check Railway dashboard for build logs" 