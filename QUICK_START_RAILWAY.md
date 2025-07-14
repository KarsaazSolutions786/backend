# Quick Start: Deploy to Railway

## 🚀 5-Minute Deployment Guide

### Step 1: Prepare Your Repository
```bash
# Make sure your code is pushed to GitHub
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### Step 2: Create Railway Project
1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository

### Step 3: Add Database
1. In your Railway project, click "New Service"
2. Select "Database" → "PostgreSQL"
3. Copy the `DATABASE_URL` from the database service

### Step 4: Configure Environment Variables
1. Go to your main service settings
2. Add these environment variables:

```env
# Database (from Step 3)
DATABASE_URL=postgresql://user:password@host:5432/database

# Security (CHANGE THESE!)
SECRET_KEY=your-super-secure-production-secret-key
JWT_SECRET=your-super-secure-production-jwt-secret

# Environment
ENVIRONMENT=production
DEBUG=false

# CORS (UPDATE WITH YOUR DOMAIN)
ALLOWED_ORIGINS=https://yourdomain.com

# AI Models
AI_MODELS_PATH=/app/models
BLOOM_MODEL_PATH=/app/models/bloom-560m
WHISPER_MODEL_PATH=/app/models/whisper-tiny.bin
SENTENCE_TRANSFORMER_MODEL_PATH=/app/models/all-MiniLM-L6-v2
XTTS_MODEL_PATH=/app/models/coqui_xtts_v2
```

### Step 5: Deploy
1. Railway will automatically detect your Dockerfile
2. Click "Deploy" to start the build
3. Wait for deployment to complete

### Step 6: Test Your Deployment
Your app will be available at: `https://your-app-name.railway.app`

Test endpoints:
- Health check: `https://your-app-name.railway.app/health`
- API docs: `https://your-app-name.railway.app/docs`

## 🔧 Troubleshooting

### Build Fails
- Check Railway build logs
- Ensure all dependencies are in `requirements.txt`
- Verify Dockerfile syntax

### Database Connection Issues
- Verify `DATABASE_URL` is correct
- Check if database service is running
- Run migrations: `railway shell` then `alembic upgrade head`

### Port Issues
- Railway automatically sets `PORT` environment variable
- Your app uses `os.getenv("PORT", "8000")`

## 📊 Monitoring

- View logs: Railway dashboard → Logs tab
- Monitor performance: Railway dashboard → Metrics tab
- Health checks: `/health` endpoint

## 🔒 Security Checklist

- [ ] Changed default JWT secrets
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configured `ALLOWED_ORIGINS`
- [ ] Enabled HTTPS (automatic with Railway)

## 💰 Cost Optimization

- Start with smaller instances
- Use Railway's managed PostgreSQL
- Monitor usage in Railway dashboard
- Scale down during development

## 🆘 Need Help?

- Railway Docs: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- Check build logs in Railway dashboard 