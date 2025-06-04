# Eindr - AI-Powered Reminder App Backend

A modular FastAPI backend for an AI-powered reminder application with speech-to-text, text-to-speech, intent classification, and conversational AI capabilities.

## 🚀 Features

- **AI-Powered Services**:

  - Speech-to-Text (Vosk)
  - Text-to-Speech (Coqui TTS)
  - Intent Classification (MiniLM)
  - Conversational AI (Bloom 560M)

- **Core Functionality**:

  - User authentication with JWT tokens
  - Reminder management with scheduling
  - Note-taking system
  - Expense tracking (ledger)
  - Friend management
  - Real-time notifications

- **API Features**:
  - RESTful API design
  - Automatic API documentation
  - File upload support
  - Audio processing
  - Background task scheduling

## 📁 Project Structure

```
eindr_backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── api/                   # API route handlers
│   ├── __init__.py
│   ├── auth.py           # Authentication endpoints
│   ├── reminders.py      # Reminder CRUD operations
│   ├── notes.py          # Note management
│   ├── ledger.py         # Expense tracking
│   ├── friends.py        # Friend management
│   ├── users.py          # User profile management
│   └── stt.py            # Speech-to-text & AI endpoints
├── core/                  # Core application logic
│   ├── config.py         # Application configuration
│   ├── security.py       # JWT & password handling
│   └── scheduler.py      # Background task scheduler
├── services/              # AI and business logic services
│   ├── stt_service.py    # Speech-to-text service
│   ├── tts_service.py    # Text-to-speech service
│   ├── intent_service.py # Intent classification
│   ├── chat_service.py   # Conversational AI
│   ├── auth_service.py   # Authentication logic
│   ├── reminder_service.py
│   ├── note_service.py
│   ├── ledger_service.py
│   ├── friend_service.py
│   └── user_service.py
└── utils/
    └── logger.py         # Logging configuration
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd eindr_backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
# App Settings
DEBUG=True
HOST=127.0.0.1
PORT=8000

# Security
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Model Paths (update these paths)
VOSK_MODEL_PATH=./models/vosk-model-en-us-0.22
TTS_MODEL_PATH=./models/tts
INTENT_MODEL_PATH=./models/intent
CHAT_MODEL_PATH=./models/bloom-560m

# Database
DATABASE_URL=postgresql://postgres:admin123@localhost:5432/eindr

```

### 3. AI Models Setup

For production use, download the required models:

```bash
# Create models directory
mkdir -p models

# Download Vosk model (example)
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip -d models/

# For demo purposes, the services use dummy implementations
```

## 🚀 Running the Application

### Development Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Production Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:

- **API**: http://127.0.0.1:8000
- **Interactive Docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📚 API Endpoints

### Authentication

- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/auth/me` - Get current user info
- `POST /api/v1/auth/refresh-token` - Refresh access token
- `POST /api/v1/auth/logout` - User logout

### Reminders

- `POST /api/v1/reminders/` - Create reminder
- `GET /api/v1/reminders/` - List reminders
- `GET /api/v1/reminders/{id}` - Get specific reminder
- `PUT /api/v1/reminders/{id}` - Update reminder
- `DELETE /api/v1/reminders/{id}` - Delete reminder
- `POST /api/v1/reminders/{id}/complete` - Mark as completed
- `GET /api/v1/reminders/upcoming/today` - Today's reminders

### Notes

- `POST /api/v1/notes/` - Create note
- `GET /api/v1/notes/` - List notes
- `GET /api/v1/notes/{id}` - Get specific note
- `PUT /api/v1/notes/{id}` - Update note
- `DELETE /api/v1/notes/{id}` - Delete note

### Speech & AI

- `POST /api/v1/stt/transcribe` - Transcribe audio file
- `POST /api/v1/stt/transcribe-and-respond` - Full AI pipeline
- `GET /api/v1/stt/response-audio/{text}` - Generate TTS audio
- `GET /api/v1/stt/voices` - Available TTS voices
- `POST /api/v1/stt/intent-classify` - Classify text intent
- `GET /api/v1/stt/intent-suggestions` - Get intent suggestions

### Other Endpoints

- `GET /` - API status
- `GET /health` - Health check with service status

## 🔐 Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer <your-jwt-token>
```

### Test User

For testing, use the pre-created user:

- **Email**: test@example.com
- **Password**: testpassword

## 🤖 AI Services

### Current Implementation

The AI services are currently implemented with dummy/mock responses for demonstration purposes. This allows the API to run without requiring large AI models.

### Production Setup

To use real AI models:

1. **Speech-to-Text (Vosk)**:

   - Download Vosk model
   - Uncomment real implementation in `services/stt_service.py`

2. **Text-to-Speech (Coqui TTS)**:

   - Install TTS models
   - Uncomment real implementation in `services/tts_service.py`

3. **Intent Classification**:

   - Train or download MiniLM model
   - Uncomment real implementation in `services/intent_service.py`

4. **Chat (Bloom 560M)**:
   - Download Bloom model
   - Uncomment real implementation in `services/chat_service.py`

## 📝 Example Usage

### Create a Reminder via API

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/reminders/" \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Call mom",
    "description": "Weekly check-in call",
    "scheduled_time": "2024-01-15T15:00:00Z"
  }'
```

### Upload Audio for Transcription

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/stt/transcribe" \
  -H "Authorization: Bearer <your-token>" \
  -F "audio_file=@recording.wav"
```

## 🔧 Configuration

Key configuration options in `core/config.py`:

- **DEBUG**: Enable debug mode
- **HOST/PORT**: Server binding
- **SECRET_KEY**: JWT signing key
- **MODEL_PATHS**: AI model locations
- **DATABASE_URL**: Database connection
- **MAX_FILE_SIZE**: Upload limit
- **AUDIO_SAMPLE_RATE**: Audio processing settings

## 📊 Monitoring & Logging

- Logs are written to `logs/` directory
- Health check endpoint: `/health`
- Service status monitoring included
- Structured logging with timestamps

## 🚀 Deployment

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```env
DEBUG=False
SECRET_KEY=your-production-secret-key
DATABASE_URL=postgresql://user:pass@localhost/eindr
ALLOWED_HOSTS=["yourdomain.com"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the API documentation at `/docs`
2. Review the logs in `logs/` directory
3. Ensure all dependencies are installed
4. Verify model paths in configuration

---

**Eindr Backend** - Making AI-powered reminders accessible and intelligent! 🤖✨

# Deploy to Railway 🚂

This section provides complete instructions for deploying your FastAPI + PostgreSQL project to Railway.

## Prerequisites

- ✅ **GitHub Repository**: Your code should be pushed to GitHub
- ✅ **Railway Account**: Create account at [railway.com](https://railway.com/)
- ✅ **All deployment files**: Listed below

## Deployment Files

Your repository should include these files (✅ already included):

```
├── Dockerfile                 # Container configuration
├── Procfile                  # Process definition
├── alembic.ini              # Database migrations config
├── alembic/
│   ├── env.py              # Migration environment
│   ├── script.py.mako      # Migration template
│   └── versions/           # Migration files
├── scripts/
│   └── init_db_railway.py  # Database initialization
├── railway.json            # Railway build config
├── railway.toml            # Railway service config
└── requirements.txt        # Python dependencies
```

## Step-by-Step Deployment

### 1. Create Railway Project

1. Go to [railway.com](https://railway.com/)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Connect your GitHub account
5. Select this repository
6. Railway will detect `Dockerfile` and start building

### 2. Add PostgreSQL Database

1. In Railway project dashboard
2. Click **"+ New Service"**
3. Select **"PostgreSQL"**
4. Railway automatically provides `DATABASE_URL` environment variable

### 3. Configure Environment Variables

In Railway project → Settings → Environment Variables, add:

#### Required Variables:
```bash
SECRET_KEY=8LrIcmpF1_QFIfGlLY6KtpvftqC4Co4mK4KyPOwrtOE
DEBUG=false
DEV_MODE=false
```

#### Auto-provided by Railway:
- `PORT` - Automatically set by Railway
- `DATABASE_URL` - Provided when PostgreSQL is added

#### Optional (Firebase Auth):
```bash
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com
```

### 4. Initialize Database

After successful deployment, run database initialization:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Link to your project
railway link

# Run database initialization
railway run python scripts/init_db_railway.py
```

Or using Railway dashboard:
1. Go to your service → **Settings** → **Service Variables**
2. Add one-time command: `python scripts/init_db_railway.py`

### 5. Verify Deployment

Your app should be accessible at: `https://your-app.railway.app`

**Health Check**: `GET https://your-app.railway.app/health`
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

**API Documentation**: `https://your-app.railway.app/docs`

## Database Migrations with Alembic

### Initial Migration

```bash
# Create initial migration
railway run alembic revision --autogenerate -m "Initial migration"

# Apply migration
railway run alembic upgrade head
```

### Ongoing Migrations

```bash
# Create new migration
railway run alembic revision --autogenerate -m "Add new table"

# Apply migrations
railway run alembic upgrade head

# Check migration status
railway run alembic current
```

## CI/CD with GitHub Integration

### Automatic Deployments

1. **Railway Dashboard** → Your Project → **Settings**
2. **Source** → **Deploy from GitHub**
3. Connect your repository
4. Set **Deploy Branch**: `main`
5. Enable **Auto-Deploy**: Every push to main triggers deployment

### Environment-Specific Deployments

Create multiple Railway services for different environments:

```bash
# Production
main branch → production.railway.app

# Staging  
develop branch → staging.railway.app
```

## Running One-off Commands

### Using Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link project
railway login
railway link

# Run commands
railway run python scripts/init_db_railway.py
railway run alembic upgrade head
railway run python -c "print('Hello Railway!')"

# Open shell
railway shell
```

### Using Railway Dashboard

1. **Project Dashboard** → **Service** → **Settings**
2. **Service** → **One-time Commands**
3. Add command: `python scripts/init_db_railway.py`
4. Click **Run**

## Troubleshooting

### 🔍 Common Issues

#### 1. Build Fails - Missing Dependencies
```bash
# Check requirements.txt includes all dependencies
pip freeze > requirements.txt

# Verify Dockerfile has system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    postgresql-client
```

#### 2. Database Connection Error
```bash
# Verify PostgreSQL service is added
railway status

# Check DATABASE_URL is set
railway run env | grep DATABASE_URL

# Test connection
railway run python -c "import os; print(os.getenv('DATABASE_URL'))"
```

#### 3. Port Binding Issues
```bash
# Ensure app binds to 0.0.0.0:${PORT}
# main.py should use:
uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
```

#### 4. Migration Errors
```bash
# Check migration status
railway run alembic current

# Force migration to head
railway run alembic stamp head

# Reset and recreate migrations
railway run python scripts/init_db_railway.py
```

#### 5. Model Download Timeout
```bash
# Pre-download models during build
# Dockerfile already includes:
RUN python download_coqui_model.py
```

#### 6. Health Check Failures on Railway
```bash
# Common causes and solutions:

# 1. App takes too long to start
# - Uses MINIMAL_MODE=true to skip AI model loading
# - Dockerfile uses requirements.railway.txt with minimal dependencies

# 2. Check Railway logs
railway logs --follow

# 3. Verify environment variables
railway run env | grep -E "(MINIMAL_MODE|PORT|DATABASE_URL)"

# 4. Test health endpoint directly
# Should return: {"status": "healthy", "environment": "railway"}
```

#### 7. MINIMAL_MODE Configuration
```bash
# For Railway deployment (automatically set in Dockerfile):
MINIMAL_MODE=true

# This disables:
# - AI model loading during startup
# - Heavy dependencies (librosa, scikit-learn, etc.)
# - Coqui STT model download

# Benefits:
# - Faster startup time (< 30 seconds)
# - Lower memory usage
# - More reliable health checks
# - Essential APIs still work (database, basic endpoints)
```

### 📊 Monitoring

#### View Logs
```bash
# Real-time logs
railway logs

# Follow logs
railway logs --follow

# Filter logs
railway logs --filter "ERROR"
```

#### Check Resources
1. **Railway Dashboard** → **Metrics**
2. Monitor CPU, Memory, Network usage
3. Set up alerts for high resource usage

### 🔧 Performance Optimization

#### Database Connection Pooling
```python
# connect_db.py already configured:
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)
```

#### Dockerfile Optimization
```dockerfile
# Multi-stage builds for smaller images
# Cache pip dependencies
# Use .dockerignore to exclude unnecessary files
```

## Advanced Configuration

### Custom Domains

1. **Railway Dashboard** → **Settings** → **Domains**
2. Add custom domain: `api.yourdomain.com`
3. Railway provides SSL certificates automatically

### Environment Variables Management

```bash
# Set environment variables via CLI
railway set SECRET_KEY=your-secret-key
railway set DEBUG=false

# Load from .env file
railway set --from-env-file .env.production
```

### Scaling

Railway automatically scales based on traffic. For manual scaling:

1. **Dashboard** → **Settings** → **Scaling**
2. Set **Memory**: 512MB - 8GB
3. Set **CPU**: 0.5 - 8 vCPU

## Useful Commands Reference

```bash
# Railway CLI Commands
railway login                    # Login to Railway
railway link                     # Link local project to Railway
railway status                   # Show project status
railway logs                     # View logs
railway run <command>            # Run one-off command
railway shell                    # Open shell in Railway environment
railway deploy                   # Manual deploy
railway set KEY=value            # Set environment variable

# Database Commands
railway run python scripts/init_db_railway.py  # Initialize database
railway run alembic upgrade head                # Run migrations
railway run alembic current                     # Check migration status
railway run python -c "from connect_db import engine; print(engine.url)"  # Test DB connection
```

## Production Checklist

- [ ] ✅ All environment variables set
- [ ] ✅ Database initialized
- [ ] ✅ Health check returns healthy
- [ ] ✅ API documentation accessible
- [ ] ✅ Custom domain configured (optional)
- [ ] ✅ Monitoring and alerts set up
- [ ] ✅ Database backups enabled
- [ ] ✅ SSL certificate active
- [ ] ✅ Auto-deployments configured

## Support

- 📚 [Railway Documentation](https://docs.railway.app/)
- 💬 [Railway Discord](https://discord.gg/railway)
- 🛠️ [Railway CLI Docs](https://docs.railway.app/reference/cli)

---

Your FastAPI + PostgreSQL app is now ready for production on Railway! 🎉

## Main API Endpoints

After deployment, these endpoints will be available:

- **Voice Pipeline**: `POST /api/v1/stt/transcribe-and-respond`
- **Health Check**: `GET /health`
- **API Docs**: `GET /docs`
- **Ledger**: `GET /api/v1/ledger/`

**Audio Requirements**: WAV format, 16kHz, mono, 16-bit PCM
