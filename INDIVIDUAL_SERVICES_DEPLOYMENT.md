# 🚀 Individual Microservices Deployment on Railway

## Overview
This guide will help you deploy each microservice separately on Railway, giving each service its own URL and independent scaling capabilities.

## 📋 Service List

| Service | Port | Database | Railway Project Name | URL Pattern |
|---------|------|----------|---------------------|-------------|
| **Auth Service** | 8001 | PostgreSQL | eindr-auth | `https://eindr-auth.railway.app` |
| **Customer Service** | 8002 | PostgreSQL | eindr-customers | `https://eindr-customers.railway.app` |
| **Reminder Service** | 8003 | PostgreSQL | eindr-reminders | `https://eindr-reminders.railway.app` |
| **Note Service** | 8004 | PostgreSQL | eindr-notes | `https://eindr-notes.railway.app` |
| **Ledger Service** | 8005 | PostgreSQL | eindr-ledger | `https://eindr-ledger.railway.app` |
| **Friend Service** | 8006 | PostgreSQL | eindr-friends | `https://eindr-friends.railway.app` |
| **History Service** | 8007 | PostgreSQL | eindr-history | `https://eindr-history.railway.app` |
| **STT Service** | 8008 | - | eindr-stt | `https://eindr-stt.railway.app` |
| **TTS Service** | 8009 | - | eindr-tts | `https://eindr-tts.railway.app` |
| **Intent Service** | 8010 | - | eindr-intent | `https://eindr-intent.railway.app` |
| **Chat Service** | 8011 | PostgreSQL | eindr-chat | `https://eindr-chat.railway.app` |
| **Scheduler Service** | 8012 | PostgreSQL | eindr-scheduler | `https://eindr-scheduler.railway.app` |
| **API Gateway** | 8080 | - | eindr-gateway | `https://eindr-gateway.railway.app` |

## 🚀 Quick Deployment

### Option 1: Automated Script
```bash
# Run the deployment script
./deploy_individual_services.sh
```

### Option 2: Manual Deployment

#### Step 1: Deploy Auth Service (Start Here)
```bash
cd services/auth-service
railway init --name eindr-auth
railway up
```

#### Step 2: Deploy Other Services
```bash
# Customer Service
cd services/customer-service
railway init --name eindr-customers
railway up

# Reminder Service
cd services/reminder-service
railway init --name eindr-reminders
railway up

# Continue for all services...
```

## ⚙️ Environment Configuration

### 1. Database Setup
For each service that needs a database:

1. Go to Railway dashboard
2. Click "New Service" → "Database" → "PostgreSQL"
3. Copy the `DATABASE_URL`
4. Add to service environment variables

### 2. Environment Variables

#### Auth Service
```env
DATABASE_URL=postgresql://user:password@host:5432/database
SECRET_KEY=your-super-secure-production-secret-key
JWT_SECRET=your-super-secure-production-jwt-secret
ENVIRONMENT=production
DEBUG=false
ALLOWED_ORIGINS=https://yourdomain.com
```

#### Customer Service
```env
DATABASE_URL=postgresql://user:password@host:5432/database
JWT_SECRET=your-super-secure-production-jwt-secret
AUTH_SERVICE_URL=https://eindr-auth.railway.app
ENVIRONMENT=production
DEBUG=false
ALLOWED_ORIGINS=https://yourdomain.com
```

#### Reminder Service
```env
DATABASE_URL=postgresql://user:password@host:5432/database
JWT_SECRET=your-super-secure-production-jwt-secret
AUTH_SERVICE_URL=https://eindr-auth.railway.app
ENVIRONMENT=production
DEBUG=false
ALLOWED_ORIGINS=https://yourdomain.com
```

#### AI Services (STT, TTS, Intent)
```env
AI_MODELS_PATH=/app/models
BLOOM_MODEL_PATH=/app/models/bloom-560m
WHISPER_MODEL_PATH=/app/models/whisper-tiny.bin
SENTENCE_TRANSFORMER_MODEL_PATH=/app/models/all-MiniLM-L6-v2
XTTS_MODEL_PATH=/app/models/coqui_xtts_v2
AUTH_SERVICE_URL=https://eindr-auth.railway.app
ENVIRONMENT=production
DEBUG=false
```

### 3. Database Migrations
After deployment, run migrations:
```bash
# Connect to service
railway shell

# Run migrations
alembic upgrade head
```

## 🔗 Service URLs After Deployment

Once deployed, your services will be available at:

```
Auth Service:      https://eindr-auth.railway.app
Customer Service:  https://eindr-customers.railway.app
Reminder Service:  https://eindr-reminders.railway.app
Note Service:      https://eindr-notes.railway.app
Ledger Service:    https://eindr-ledger.railway.app
Friend Service:    https://eindr-friends.railway.app
History Service:   https://eindr-history.railway.app
STT Service:       https://eindr-stt.railway.app
TTS Service:       https://eindr-tts.railway.app
Intent Service:    https://eindr-intent.railway.app
Chat Service:      https://eindr-chat.railway.app
Scheduler Service: https://eindr-scheduler.railway.app
API Gateway:       https://eindr-gateway.railway.app
```

## 🧪 Testing Individual Services

### Health Checks
```bash
# Test each service health
curl https://eindr-auth.railway.app/health
curl https://eindr-customers.railway.app/health
curl https://eindr-reminders.railway.app/health
# ... etc
```

### API Documentation
```bash
# Access Swagger docs for each service
https://eindr-auth.railway.app/docs
https://eindr-customers.railway.app/docs
https://eindr-reminders.railway.app/docs
# ... etc
```

### Authentication Flow
```bash
# 1. Register user
curl -X POST https://eindr-auth.railway.app/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# 2. Login
curl -X POST https://eindr-auth.railway.app/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# 3. Use token to access other services
curl -X GET https://eindr-customers.railway.app/customers \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🔧 API Gateway Configuration

After deploying all services, update your API Gateway environment variables:

```env
# Service URLs
AUTH_SERVICE_URL=https://eindr-auth.railway.app
CUSTOMER_SERVICE_URL=https://eindr-customers.railway.app
REMINDER_SERVICE_URL=https://eindr-reminders.railway.app
NOTE_SERVICE_URL=https://eindr-notes.railway.app
LEDGER_SERVICE_URL=https://eindr-ledger.railway.app
FRIEND_SERVICE_URL=https://eindr-friends.railway.app
HISTORY_SERVICE_URL=https://eindr-history.railway.app
STT_SERVICE_URL=https://eindr-stt.railway.app
TTS_SERVICE_URL=https://eindr-tts.railway.app
INTENT_SERVICE_URL=https://eindr-intent.railway.app
CHAT_SERVICE_URL=https://eindr-chat.railway.app
SCHEDULER_SERVICE_URL=https://eindr-scheduler.railway.app
```

## 📊 Monitoring and Scaling

### Individual Service Monitoring
- Each service has its own Railway dashboard
- Monitor logs, metrics, and performance
- Set up alerts for each service

### Scaling
```bash
# Scale individual services based on demand
# Go to Railway dashboard for each service
# Adjust instance count and resources
```

### Health Monitoring
```bash
# Check all services health
for service in eindr-auth eindr-customers eindr-reminders eindr-notes eindr-ledger eindr-friends eindr-history eindr-stt eindr-tts eindr-intent eindr-chat eindr-scheduler; do
  echo "Checking $service..."
  curl -s https://$service.railway.app/health
  echo ""
done
```

## 🛠️ Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Railway build logs
   - Ensure all dependencies are in `requirements.txt`
   - Verify Dockerfile syntax

2. **Database Connection Issues**
   - Verify `DATABASE_URL` is correct
   - Check if database service is running
   - Run migrations: `railway shell && alembic upgrade head`

3. **Service Communication Issues**
   - Verify service URLs are correct
   - Check CORS configuration
   - Ensure JWT secrets match across services

4. **Port Issues**
   - Railway automatically sets the `PORT` environment variable
   - Your app should use `os.getenv("PORT", "8000")`

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

## 💰 Cost Optimization

1. **Start Small**: Begin with smaller instance types
2. **Scale Up**: Increase resources based on actual usage
3. **Database Optimization**: Use Railway's managed PostgreSQL
4. **Caching**: Implement Redis caching to reduce database calls
5. **Monitoring**: Use Railway's built-in monitoring to optimize costs

## 🔒 Security Considerations

1. **Environment Variables**: Never commit sensitive data to Git
2. **JWT Secrets**: Use strong, unique secrets for production
3. **CORS**: Configure `ALLOWED_ORIGINS` for production
4. **HTTPS**: Railway provides automatic HTTPS
5. **Database**: Use Railway's managed PostgreSQL for security

## 📝 Next Steps

1. Deploy all services using the script or manual process
2. Configure environment variables for each service
3. Run database migrations
4. Test each service individually
5. Configure API Gateway with service URLs
6. Set up monitoring and alerting
7. Implement CI/CD pipelines
8. Configure custom domains if needed

## 🆘 Support

- Railway Documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- GitHub Issues: Create issues in your repository for code-specific problems 