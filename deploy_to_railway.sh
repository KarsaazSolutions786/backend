#!/bin/bash

# Railway Deployment Script
# This script commits and pushes the Railway deployment fixes

echo "🚂 Deploying Railway fixes..."

# Add all changes
git add .

# Commit with descriptive message
git commit -m "Fix Railway deployment: add MINIMAL_MODE, update health checks, use minimal requirements

- Add MINIMAL_MODE to skip AI service initialization on Railway
- Update Dockerfile to use requirements.railway.txt with minimal dependencies
- Fix health check endpoint to work without AI services
- Update startup sequence to be more fault-tolerant
- Add Railway environment detection
- Improve error handling and logging
- Add troubleshooting documentation for health check failures"

# Push to main branch
git push origin main

echo "✅ Changes pushed to GitHub!"
echo "📡 Railway will automatically detect and redeploy..."
echo "🔍 Monitor deployment at: https://railway.app"
echo "⏱️ Expected startup time: < 60 seconds"
echo ""
echo "Next steps:"
echo "1. Check Railway logs: railway logs --follow"
echo "2. Verify health check: https://your-app.railway.app/health"
echo "3. Test API docs: https://your-app.railway.app/docs" 