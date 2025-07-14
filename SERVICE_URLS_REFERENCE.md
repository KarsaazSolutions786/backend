# 🔗 Service URLs Reference Card

## 🚀 Individual Service URLs

After deployment, your services will be available at these URLs:

### 🔐 Authentication & User Management
- **Auth Service**: `https://eindr-auth.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Register: `/auth/register`
  - Login: `/auth/login`
  - Refresh: `/auth/refresh`

### 👥 Customer Management
- **Customer Service**: `https://eindr-customers.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Customers: `/customers`

### ⏰ Reminders & Scheduling
- **Reminder Service**: `https://eindr-reminders.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Reminders: `/reminders`

- **Scheduler Service**: `https://eindr-scheduler.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Jobs: `/jobs`

### 📝 Content Management
- **Note Service**: `https://eindr-notes.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Notes: `/notes`

### 💰 Financial Management
- **Ledger Service**: `https://eindr-ledger.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Expenses: `/expenses`

### 👥 Social Features
- **Friend Service**: `https://eindr-friends.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Friends: `/friends`

### 📊 Activity & History
- **History Service**: `https://eindr-history.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Logs: `/logs`

### 🤖 AI Services
- **STT Service**: `https://eindr-stt.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Transcribe: `/transcribe`

- **TTS Service**: `https://eindr-tts.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Synthesize: `/synthesize`

- **Intent Service**: `https://eindr-intent.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Classify: `/classify`

- **Chat Service**: `https://eindr-chat.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Conversations: `/conversations`

### 🌐 API Gateway
- **Main Gateway**: `https://eindr-gateway.railway.app`
  - Health: `/health`
  - Docs: `/docs`
  - Status: `/api/status`

## 🧪 Quick Testing Commands

### Health Check All Services
```bash
# Test all services health
curl https://eindr-auth.railway.app/health
curl https://eindr-customers.railway.app/health
curl https://eindr-reminders.railway.app/health
curl https://eindr-notes.railway.app/health
curl https://eindr-ledger.railway.app/health
curl https://eindr-friends.railway.app/health
curl https://eindr-history.railway.app/health
curl https://eindr-stt.railway.app/health
curl https://eindr-tts.railway.app/health
curl https://eindr-intent.railway.app/health
curl https://eindr-chat.railway.app/health
curl https://eindr-scheduler.railway.app/health
curl https://eindr-gateway.railway.app/health
```

### Authentication Flow
```bash
# 1. Register
curl -X POST https://eindr-auth.railway.app/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# 2. Login
curl -X POST https://eindr-auth.railway.app/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# 3. Use token (replace YOUR_TOKEN_HERE)
curl -X GET https://eindr-customers.railway.app/customers \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### API Documentation URLs
```bash
# Open these in your browser to see API docs
open https://eindr-auth.railway.app/docs
open https://eindr-customers.railway.app/docs
open https://eindr-reminders.railway.app/docs
open https://eindr-notes.railway.app/docs
open https://eindr-ledger.railway.app/docs
open https://eindr-friends.railway.app/docs
open https://eindr-history.railway.app/docs
open https://eindr-stt.railway.app/docs
open https://eindr-tts.railway.app/docs
open https://eindr-intent.railway.app/docs
open https://eindr-chat.railway.app/docs
open https://eindr-scheduler.railway.app/docs
open https://eindr-gateway.railway.app/docs
```

## 🔧 Environment Variables for API Gateway

After deploying all services, update your API Gateway with these URLs:

```env
# Service URLs for API Gateway
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

## 📊 Monitoring Dashboard URLs

Each service has its own Railway dashboard:
- Go to [railway.app](https://railway.app)
- Select your project
- View logs, metrics, and deployment status

## 🚨 Emergency Contacts

- **Railway Support**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **Service Status**: Check individual service dashboards 