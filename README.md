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
DATABASE_URL=sqlite:///./eindr.db
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
