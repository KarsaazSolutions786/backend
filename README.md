# Eindr - AI-Powered Reminder App Backend

A sophisticated FastAPI backend for an AI-powered reminder application featuring speech-to-text, text-to-speech, intent classification, conversational AI, and comprehensive data management capabilities.

## 🚀 Features

- **AI-Powered Services**:
  - Speech-to-Text (Whisper STT)
  - Text-to-Speech (Coqui TTS)
  - Intent Classification (MiniLM, PyTorch)
  - Conversational AI (Bloom 560M)
  - Multi-intent processing

- **Core Functionality**:
  - User authentication with JWT tokens
  - Reminder management with scheduling
  - Note-taking system
  - Expense tracking (ledger)
  - Friend management
  - Real-time notifications
  - Admin panel with comprehensive dashboard

- **API Features**:
  - RESTful API design
  - Automatic API documentation
  - File upload support
  - Audio processing
  - Background task scheduling
  - Complete AI pipeline integration

## 🏗️ System Architecture & Complete Workflow

### Core Workflow Overview

The Eindr backend follows a sophisticated AI-driven pipeline that processes user input through multiple stages:

```
1. Audio Input → 2. Speech-to-Text → 3. Intent Classification → 4. Database Operations → 5. Response Generation → 6. Text-to-Speech → 7. Audio Response
```

### 1. Application Startup & Service Initialization

**Entry Point: `main.py`**

The application starts with intelligent service initialization:

```python
# Lifespan manager handles startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Environment Detection
    is_minimal_mode = os.getenv("MINIMAL_MODE", "false").lower() == "true"
    is_railway_env = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    # 2. Service Selection (Adaptive based on environment)
    if is_minimal_mode or is_railway_env:
        # Lightweight services for production/Railway
        intent_service = IntentService()  # MiniLM-based
    else:
        # Full services for development
        intent_service = PyTorchIntentService()  # Advanced PyTorch models
    
    # 3. Core Service Initialization
    stt_service = SpeechToTextService()      # Whisper STT
    tts_service = TextToSpeechService()      # Coqui TTS
    chat_service = ChatService()             # Bloom 560M
    
    # 4. Service Registration
    set_services(stt_service, tts_service, intent_service, chat_service)
```

**Service Architecture:**
- **Adaptive Loading**: Different service configurations for development vs production
- **Graceful Fallbacks**: If advanced services fail, lightweight alternatives are used
- **Memory Management**: PyTorch memory optimization for GPU environments

### 2. Complete AI Pipeline Workflow

**Main Pipeline: `services/ai_pipeline_service.py`**

The AI Pipeline Service coordinates the entire workflow:

#### Stage 1: Speech-to-Text Processing
```python
# Input: Audio file path
stt_result = await self.whisper_stt.transcribe_audio(audio_file_path, language)
# Output: Transcribed text with confidence scores
```

#### Stage 2: Intent Classification
```python
# Input: Transcribed text
intent_result = await self.minilm_intent.classify_intent(transcription, multi_intent)
# Output: Intent(s) with confidence scores

# Supported Intents:
# - create_reminder, update_reminder, delete_reminder
# - create_note, get_notes
# - ledger_operations (owe/owed tracking)
# - friend_management
# - chit_chat, general_query
```

#### Stage 3: Database Operations (Intent-Based Routing)
```python
# Route based on detected intent
if intent in ["chit_chat", "general_query"]:
    # Route to Conversational AI
    chat_response = await self.chat_service.generate_response(
        message=transcription, user_id=user_id, context=chat_context
    )
else:
    # Route to Database Operations
    db_result = await self.database_service.process_intent(
        intent_result, user_id
    )
```

#### Stage 4: Response Generation & Text-to-Speech
```python
# Generate appropriate response text
response_text = self._generate_response_text(intent_result, db_result)

# Convert to speech
audio_data = await self.coqui_tts.synthesize_speech(response_text, voice)
```

### 3. Database Architecture & Models

**Database Schema: `models/models.py`**

The system uses a comprehensive PostgreSQL schema:

```sql
-- Core User Management
users: id, email, password_hash, language, timezone, created_at
preferences: user_id, allow_friends, notification_sound, tts_language

-- Content Management
reminders: id, user_id, title, description, time, repeat_pattern, is_shared
notes: id, user_id, content, source, created_at
ledger_entries: id, user_id, contact_name, amount, direction (owe/owed)

-- Social Features
friendships: id, user_id, friend_id, status (pending/accepted/blocked)
permissions: id, user_id, friend_id, auto_accept_reminders, auto_accept_notes

-- AI & Analytics
embeddings: id, user_id, reminder_id, embedding (vector data)
history_logs: id, user_id, content, interaction_type, created_at
```

### 4. API Endpoint Structure & Data Flow

**Authentication Flow:**
```
POST /api/v1/auth/register → User Creation → JWT Token Generation
POST /api/v1/auth/login → Credential Validation → JWT Token
GET /api/v1/auth/me → Token Validation → User Profile
```

**AI Pipeline Endpoints:**
```
POST /api/v1/stt/transcribe-and-respond
├── Audio Upload & Validation
├── Complete AI Pipeline Processing
├── Database Operations
├── TTS Response Generation
└── Return JSON + Audio Response

POST /api/v1/ai-pipeline/process-complete
├── Full Pipeline with Configuration Options
├── Multi-intent Support
├── Custom Voice Selection
└── Pipeline Statistics
```

**CRUD Operations:**
```
Reminders: POST, GET, PUT, DELETE /api/v1/reminders/
├── Create with natural language parsing
├── Update with conflict resolution
├── Delete with cascade handling
└── List with filtering & pagination

Notes: POST, GET, PUT, DELETE /api/v1/notes/
Ledger: POST, GET, PUT, DELETE /api/v1/ledger/
Friends: POST, GET, PUT, DELETE /api/v1/friends/
```

### 5. Service Layer Architecture

**Service Responsibilities:**

1. **STT Services** (`services/whisper_stt_service.py`)
   - Audio preprocessing and validation
   - Whisper model integration
   - Language detection
   - Confidence scoring

2. **Intent Services** (Multiple implementations)
   - `minilm_intent_service.py`: Lightweight MiniLM-based classification
   - `pytorch_intent_service.py`: Advanced PyTorch models
   - Multi-intent detection and confidence scoring

3. **Database Integration** (`services/database_integration_service.py`)
   - Intent-to-database operation mapping
   - Transaction management
   - Error handling and rollback

4. **Chat Service** (`services/chat_service.py`)
   - Bloom 560M model integration
   - Context management
   - Conversation history

5. **TTS Services** (`services/coqui_tts_service.py`)
   - Coqui TTS model integration
   - Voice selection and customization
   - Audio format optimization

### 6. Configuration & Environment Management

**Configuration System: `core/config.py`**

Environment-aware configuration:
```python
# Railway Production Environment
IS_RAILWAY: bool = os.getenv("RAILWAY_ENVIRONMENT") is not None
MINIMAL_MODE: bool = os.getenv("MINIMAL_MODE", "false").lower() == "true"

# AI Model Paths (Environment-specific)
CHAT_MODEL_NAME: str = "bigscience/bloom-560m"
VLLM_SERVER_URL: str = "http://localhost:8001"

# Performance Tuning
VLLM_GPU_MEMORY_UTILIZATION: float = 0.8
VLLM_MAX_MODEL_LEN: int = 2048
```

### 7. Admin Panel & System Management

**Admin Features:**
- User management and analytics
- System health monitoring
- Pipeline performance metrics
- Database administration
- Service status monitoring

**Admin Endpoints:**
```
POST /api/v1/admin/login → Admin Authentication
GET /api/v1/admin/dashboard → System Statistics
GET /api/v1/admin/users → User Management
GET /api/v1/admin/users/{id}/stats → User Analytics
```

### 8. Error Handling & Fallback Mechanisms

**Graceful Degradation:**
1. **Service Failures**: Automatic fallback to lighter alternatives
2. **Model Loading**: Fallback from PyTorch to MiniLM if memory insufficient
3. **TTS Failures**: Fallback to GTTS if Coqui fails
4. **Database Errors**: Transaction rollback with user-friendly messages

### 9. Performance Optimization

**Memory Management:**
- PyTorch memory optimization for GPU environments
- Model lazy loading based on usage
- Connection pooling for database operations

**Caching:**
- KPI caching system (`utils/kpi_cache.py`)
- Model prediction caching
- Response caching for common queries

**Background Processing:**
- Scheduler for reminder notifications
- Async processing for heavy AI operations
- Queue management for concurrent requests

## 📁 Project Structure

```
eindr_backend/
├── main.py                          # FastAPI app + service initialization
├── core/                           # Core application logic
│   ├── config.py                   # Environment-aware configuration
│   ├── security.py                 # JWT & authentication
│   ├── scheduler.py                # Background task scheduler
│   ├── dependencies.py             # Dependency injection
│   └── torch_config.py             # PyTorch optimization
├── api/                            # API route handlers
│   ├── auth.py                     # Authentication endpoints
│   ├── ai_pipeline.py              # AI pipeline endpoints
│   ├── stt.py                      # Speech & AI processing
│   ├── reminders.py                # Reminder CRUD
│   ├── notes.py                    # Note management
│   ├── ledger.py                   # Expense tracking
│   ├── friends.py                  # Social features
│   └── users.py                    # User management
├── services/                       # Business logic & AI services
│   ├── ai_pipeline_service.py      # Main AI workflow coordinator
│   ├── whisper_stt_service.py      # Speech-to-text
│   ├── minilm_intent_service.py    # Intent classification (lightweight)
│   ├── pytorch_intent_service.py   # Intent classification (advanced)
│   ├── coqui_tts_service.py        # Text-to-speech
│   ├── chat_service.py             # Conversational AI
│   ├── database_integration_service.py # Database operations
│   └── [other services]
├── models/                         # Database models
│   ├── models.py                   # SQLAlchemy models
│   └── admin_models.py             # Admin-specific models
├── routers/admin/                  # Admin panel routes
│   ├── auth.py                     # Admin authentication
│   ├── dashboard.py                # Admin dashboard
│   └── users.py                    # User management
├── utils/                          # Utilities
│   ├── logger.py                   # Logging configuration
│   └── kpi_cache.py                # Performance caching
└── alembic/                        # Database migrations
    ├── versions/                   # Migration files
    └── env.py                      # Migration configuration
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database
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
HOST=0.0.0.0
PORT=8000
MINIMAL_MODE=false  # Set to true for lightweight deployment

# Security
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql://postgres:admin123@localhost:5432/eindr

# AI Models Configuration
CHAT_MODEL_NAME=bigscience/bloom-560m
VLLM_SERVER_URL=http://localhost:8001
CHAT_MAX_TOKENS=150
CHAT_TEMPERATURE=0.7

# Performance Tuning
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_MAX_MODEL_LEN=2048
```

### 3. Database Setup

```bash
# Initialize database
python init_db.py

# Run migrations
alembic upgrade head

# Seed admin user (optional)
python scripts/seed_admin.py
```

### 4. AI Models Setup (For Full Mode)

```bash
# Download required models
python download_coqui_model.py

# Train intent models (optional)
python train_intent_model.py
```

## 🚀 Running the Application

### Development Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Server (Railway/Minimal Mode)

```bash
# Set environment variables
export MINIMAL_MODE=true
export RAILWAY_ENVIRONMENT=production

# Run with optimized settings
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

The API will be available at:

- **API**: http://0.0.0.0:8000
- **Interactive Docs**: http://0.0.0.0:8000/docs
- **ReDoc**: http://0.0.0.0:8000/redoc

## 📚 API Workflow Examples

### Complete AI Pipeline Workflow

```bash
# 1. Upload audio and process through complete pipeline
curl -X POST "http://localhost:8000/api/v1/stt/transcribe-and-respond" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@reminder.wav" \
  -F "language=en" \
  -F "multi_intent=true" \
  -F "generate_audio_response=true"

# Response includes:
# - Transcribed text
# - Detected intent(s)
# - Database operation results
# - Generated response text
# - Audio response data
```

### Intent-Based Workflows

```bash
# Create Reminder via Voice
"Set a reminder to call mom at 5 PM tomorrow"
→ Intent: create_reminder
→ Database: INSERT INTO reminders (title, time, user_id)
→ Response: "I've set a reminder to call mom at 5 PM tomorrow"

# Add Note via Voice
"Note that I need to buy groceries"
→ Intent: create_note
→ Database: INSERT INTO notes (content, user_id)
→ Response: "I've saved your note about buying groceries"

# Track Expense via Voice
"I owe John 50 dollars for dinner"
→ Intent: ledger_operation
→ Database: INSERT INTO ledger_entries (contact_name, amount, direction)
→ Response: "I've recorded that you owe John $50 for dinner"

# Conversational Query
"How's the weather today?"
→ Intent: general_query
→ Chat Service: Generate conversational response
→ Response: AI-generated contextual response
```

## 📊 Monitoring & Analytics

### System Health Endpoints

```bash
# Basic health check
GET /health
→ Returns system status and environment info

# Detailed service status
GET /api/v1/ai-pipeline/status
→ Returns individual service health and performance metrics
```

### Admin Panel Analytics

```bash
# Admin login
POST /api/v1/admin/login
{
  "username": "admin",
  "password": "admin_password"
}

# Dashboard statistics
GET /api/v1/admin/dashboard
→ Returns user analytics, system performance, and usage statistics

# User management
GET /api/v1/admin/users
→ Returns user list with activity metrics
```

## 🔐 Authentication & Security

The API uses JWT (JSON Web Tokens) for authentication:

```bash
# Register new user
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "password": "secure_password"
}

# Login and get token
POST /api/v1/auth/login
{
  "email": "user@example.com",
  "password": "secure_password"
}

# Use token in subsequent requests
Authorization: Bearer <your-jwt-token>
```

## 🔧 Deployment Options

### Railway Deployment (Recommended for Production)

```bash
# Configure for Railway
export MINIMAL_MODE=true
export RAILWAY_ENVIRONMENT=production

# Deploy using Railway CLI
railway deploy
```

### Docker Deployment

```bash
# Build image
docker build -t eindr-backend .

# Run container
docker run -p 8000:8000 -e MINIMAL_MODE=true eindr-backend
```

### Local Development

```bash
# Full feature mode with all AI capabilities
export MINIMAL_MODE=false
python main.py
```

## 🧪 Testing

```bash
# Run test suite
python -m pytest tests/

# Test specific components
python test_minilm_intent_mapping.py
python scripts/test_chat_model.py
```

## 📈 Performance Considerations

- **Memory Usage**: Minimal mode uses ~500MB RAM vs ~2GB for full mode
- **Response Time**: AI pipeline processing typically 2-5 seconds
- **Concurrent Users**: Supports 50+ concurrent users in production
- **Database Performance**: Optimized with proper indexing and connection pooling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

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
