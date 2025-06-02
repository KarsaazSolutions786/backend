# 🚀 Railway Deployment Guide

This guide will help you deploy your Eindr Backend to [Railway](https://railway.com/).

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be pushed to a GitHub repository
2. **Railway Account**: Create an account at [railway.com](https://railway.com/)
3. **Firebase Project**: If using Firebase authentication

## 🛠️ Pre-Deployment Setup

### 1. Verify Configuration Files

Make sure these files are in your repository:
- ✅ `Dockerfile` - Container configuration
- ✅ `railway.json` - Railway build configuration
- ✅ `railway.toml` - Railway service configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `railway.env.example` - Environment variables template

### 2. Update .gitignore

Ensure sensitive files are ignored:
```
.env
firebase-service-account.json
*.log
__pycache__/
.venv/
models/
uploads/
```

## 🚂 Railway Deployment Steps

### Step 1: Create New Project

1. Go to [railway.com](https://railway.com/)
2. Click **"Deploy a new project"**
3. Select **"Deploy from GitHub Repo"**
4. Connect your GitHub account if not already connected
5. Select your repository

### Step 2: Add PostgreSQL Database

1. In your Railway project dashboard
2. Click **"+ New Service"**
3. Select **"PostgreSQL"**
4. Railway will automatically provide `DATABASE_URL` environment variable

### Step 3: Configure Environment Variables

In your Railway project settings, add these environment variables:

#### Required Variables:
```bash
DEBUG=false
DEV_MODE=false
SECRET_KEY=your-super-secret-key-change-this
```

#### Firebase Variables (if using Firebase Auth):
```bash
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your-client-id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
```

#### Auto-provided by Railway:
- `PORT` - Automatically set by Railway
- `DATABASE_URL` - Automatically provided when you add PostgreSQL

### Step 4: Deploy

1. Railway will automatically detect your `Dockerfile`
2. Click **"Deploy"** or push to your main branch
3. Railway will build and deploy your application
4. Monitor the build logs for any issues

### Step 5: Configure Domain (Optional)

1. In Railway project settings
2. Go to **"Custom Domain"**
3. Add your custom domain or use the provided Railway domain

## 🔍 Post-Deployment Verification

### Health Check
Your app should be accessible at: `https://your-app.railway.app/health`

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "stt": true,
    "tts": true,
    "intent": true,
    "chat": true
  }
}
```

### API Documentation
Access Swagger docs at: `https://your-app.railway.app/docs`

### Test Endpoints
- **Root**: `GET https://your-app.railway.app/`
- **STT Pipeline**: `POST https://your-app.railway.app/api/v1/stt/transcribe-and-respond`
- **Ledger**: `GET https://your-app.railway.app/api/v1/ledger/`

## 🐛 Troubleshooting

### Common Issues:

1. **Build Fails - Missing Dependencies**
   - Check `requirements.txt` is complete
   - Verify Dockerfile system dependencies

2. **Database Connection Error**
   - Ensure PostgreSQL service is added
   - Check `DATABASE_URL` environment variable

3. **Model Download Fails**
   - Large models may timeout during build
   - Consider pre-built Docker image with models

4. **Port Binding Issues**
   - Ensure your app binds to `0.0.0.0:${PORT}`
   - Railway automatically sets `PORT` environment variable

### View Logs:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# View logs
railway logs
```

## 📚 Useful Railway Features

- **Auto-deployments**: Pushes to main branch auto-deploy
- **Environment management**: Easy environment variable management
- **Monitoring**: Built-in metrics and logging
- **Scaling**: Automatic scaling based on demand
- **Custom domains**: Free SSL certificates
- **Team collaboration**: Share projects with team members

## 🔗 Links

- [Railway Documentation](https://docs.railway.app/)
- [Railway CLI](https://docs.railway.app/reference/cli)
- [Railway Discord](https://discord.gg/railway)

## 💡 Tips

1. **Use environment variables** for all configuration
2. **Monitor resource usage** in Railway dashboard
3. **Set up monitoring** for production apps
4. **Use Railway's preview deployments** for testing
5. **Consider using Railway's Redis** for caching if needed

---

Happy deploying! 🎉 