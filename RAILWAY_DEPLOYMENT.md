# Railway Deployment Guide for Eindr Microservices

## Overview
This guide will help you deploy your Eindr microservices backend to Railway, a modern deployment platform that supports Docker containers and provides managed databases.

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Railway CLI** (optional): Install with `npm i -g @railway/cli`

## Step 1: Prepare Your Repository

### 1.1 Update .gitignore
Make sure your `.gitignore` excludes sensitive files:
```
.env
*.log
__pycache__/
*.pyc
.DS_Store
models/
```

### 1.2 Environment Variables
Create a `railway.env` file (don't commit this) with your production environment variables:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/database

# JWT Configuration
SECRET_KEY=your-super-secure-production-secret-key
JWT_SECRET=your-super-secure-production-jwt-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Environment Configuration
ENVIRONMENT=production
DEBUG=false

# CORS Configuration
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Redis Configuration
REDIS_URL=redis://host:port

# AI Model Configuration
AI_MODELS_PATH=/app/models
BLOOM_MODEL_PATH=/app/models/bloom-560m
WHISPER_MODEL_PATH=/app/models/whisper-tiny.bin
SENTENCE_TRANSFORMER_MODEL_PATH=/app/models/all-MiniLM-L6-v2
XTTS_MODEL_PATH=/app/models/coqui_xtts_v2

# Service Configuration
SERVICE_NAME=eindr-microservice
SERVICE_VERSION=1.0.0
```

## Step 2: Deploy to Railway

### Option A: Using Railway Dashboard (Recommended)

1. **Connect Repository**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Configure Environment**:
   - Go to your project settings
   - Add all environment variables from `railway.env`
   - Set `PORT` to `8000` (Railway will override this)

3. **Deploy**:
   - Railway will automatically detect the Dockerfile
   - Click "Deploy" to start the build process

### Option B: Using Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Link to your Railway project
railway link

# Deploy
railway up
```

## Step 3: Set Up Database

### 3.1 Add PostgreSQL Service
1. In your Railway project, click "New Service"
2. Select "Database" → "PostgreSQL"
3. Railway will provide you with a `DATABASE_URL`

### 3.2 Update Environment Variables
Add the `DATABASE_URL` to your service environment variables.

### 3.3 Run Migrations
After deployment, you'll need to run database migrations:

```bash
# Connect to your Railway service
railway shell

# Run migrations
alembic upgrade head
```

## Step 4: Set Up Redis (Optional)

If you need Redis for caching/rate limiting:
1. Add a new service in Railway
2. Select "Database" → "Redis"
3. Add the `REDIS_URL` to your environment variables

## Step 5: Configure Custom Domain (Optional)

1. Go to your Railway project settings
2. Click "Domains"
3. Add your custom domain
4. Update your DNS settings as instructed

## Step 6: Monitor and Scale

### 6.1 Health Checks
Your application includes health checks at `/health` endpoint.

### 6.2 Logs
View logs in the Railway dashboard or using CLI:
```bash
railway logs
```

### 6.3 Scaling
Railway allows you to scale your service:
- Go to your service settings
- Adjust the number of instances
- Set resource limits

## Step 7: Environment-Specific Configurations

### Production Environment
```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
HSTS_ENABLED=true
CONTENT_SECURITY_POLICY_ENABLED=true
```

### Staging Environment
```env
ENVIRONMENT=staging
DEBUG=true
LOG_LEVEL=DEBUG
```

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check the build logs in Railway dashboard
   - Ensure all dependencies are in `requirements.txt`
   - Verify Dockerfile syntax

2. **Database Connection Issues**:
   - Verify `DATABASE_URL` is correct
   - Check if database service is running
   - Ensure migrations are applied

3. **Port Issues**:
   - Railway automatically sets the `PORT` environment variable
   - Your app should use `os.getenv("PORT", "8000")`

4. **Memory Issues**:
   - AI models can be memory-intensive
   - Consider using Railway's larger instance types
   - Implement model caching strategies

### Debug Commands

```bash
# Check service status
railway status

# View logs
railway logs

# Connect to service shell
railway shell

# Check environment variables
railway variables
```

## Security Considerations

1. **Environment Variables**: Never commit sensitive data to Git
2. **CORS**: Configure `ALLOWED_ORIGINS` for production
3. **JWT Secrets**: Use strong, unique secrets for production
4. **Database**: Use Railway's managed PostgreSQL for security
5. **HTTPS**: Railway provides automatic HTTPS

## Cost Optimization

1. **Instance Size**: Start with smaller instances and scale up as needed
2. **Database**: Use Railway's PostgreSQL for managed database
3. **Storage**: Consider external storage for large model files
4. **Caching**: Implement Redis caching to reduce database calls

## Next Steps

1. Set up CI/CD pipeline with GitHub Actions
2. Configure monitoring and alerting
3. Set up backup strategies
4. Implement proper logging and error tracking
5. Consider using Railway's preview deployments for testing

## Support

- Railway Documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- GitHub Issues: Create issues in your repository for code-specific problems 