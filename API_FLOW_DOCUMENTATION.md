# 🚀 Eindr Backend - Complete API Flow Documentation

## 📋 Project Overview

**Eindr** is an AI-powered reminder application backend built with **FastAPI**, featuring comprehensive voice-to-database processing, intelligent intent classification, and multi-modal AI services.

### 🏗️ Technology Stack
- **Backend Framework**: FastAPI
- **Database**: PostgreSQL 
- **Authentication**: Firebase Auth
- **AI Services**: Speech-to-Text, Text-to-Speech, Intent Classification, Chat
- **ORM**: SQLAlchemy
- **Task Scheduling**: APScheduler

---

## 🔄 System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Client App        │    │   FastAPI Backend   │    │   PostgreSQL DB     │
│   ├── Voice Input   │───▶│   ├── STT Service   │───▶│   ├── Users         │
│   ├── Text Input    │    │   ├── Intent AI     │    │   ├── Reminders     │
│   └── API Calls     │◀───│   ├── TTS Service   │◀───│   ├── Notes         │
└─────────────────────┘    │   └── Chat AI       │    │   └── Ledger        │
                           └─────────────────────┘    └─────────────────────┘
                                      │
                           ┌─────────────────────┐
                           │   Firebase Auth     │
                           │   ├── JWT Tokens    │
                           │   └── User Management│
                           └─────────────────────┘
```

---

## 📡 API Endpoints Structure

### Base URL: `http://localhost:8000/api/v1/`

| Module | Prefix | Description | Key Endpoints |
|--------|--------|-------------|---------------|
| **Authentication** | `/auth` | User registration & login | `POST /register`, `POST /login`, `POST /firebase/register` |
| **Users** | `/users` | User profile management | `GET /me`, `PUT /profile`, `GET /preferences` |
| **Reminders** | `/reminders` | Reminder CRUD operations | `POST /`, `GET /`, `GET /{id}`, `PUT /{id}`, `DELETE /{id}` |
| **Notes** | `/notes` | Note management | `POST /`, `GET /`, `GET /{id}`, `PUT /{id}`, `DELETE /{id}` |
| **Ledger** | `/ledger` | Expense/money tracking | `POST /`, `GET /`, `GET /summary`, `DELETE /{id}` |
| **Friends** | `/friends` | Friend system & sharing | `POST /invite`, `GET /`, `PUT /{id}/accept` |
| **STT** | `/stt` | Speech-to-text processing | `POST /transcribe`, `POST /transcribe-and-respond`, `GET /model-info`, `GET /voices` |
| **Intent Processor** | `/intent-processor` | Intent classification & DB storage | `POST /process`, `GET /supported-intents` |
| **Embeddings** | `/embeddings` | Vector search capabilities | `POST /generate`, `POST /search` |
| **History** | `/history` | User interaction logs | `GET /`, `GET /search` |

---

## 🎯 Core Voice-to-Database Pipeline

### Complete Flow: Voice → Database

```mermaid
graph TD
    A[🎤 User Records Audio] --> B[📤 Upload WAV File]
    B --> C[🔊 POST /stt/transcribe]
    C --> D[🤖 Speech-to-Text Processing]
    D --> E[📝 Text Transcription]
    E --> F[🧠 Intent Classification]
    F --> G[🎯 Intent + Entities + Confidence]
    G --> H[⚡ POST /intent-processor/process]
    H --> I{🔍 Intent Type?}
    
    I -->|create_reminder| J[⏰ Extract Time & Person]
    I -->|create_note| K[📋 Extract Content]
    I -->|create_ledger| L[💰 Extract Amount & Contact]
    I -->|chit_chat| M[💬 Log Interaction]
    
    J --> N[💾 Save to reminders table]
    K --> O[💾 Save to notes table]
    L --> P[💾 Save to ledger_entries table]
    M --> Q[💾 Save to history_logs table]
    
    N --> R[✅ Return Success Response]
    O --> R
    P --> R
    Q --> R
```

### Example API Call Sequence

```bash
# Step 1: Upload audio for transcription
curl -X POST "http://localhost:8000/api/v1/stt/transcribe" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -F "audio_file=@reminder.wav"

# Step 2: Process intent and save to database
curl -X POST "http://localhost:8000/api/v1/intent-processor/process" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "intent_data": {
      "intent": "create_reminder",
      "confidence": 0.95,
      "entities": {"time": ["5", "00", "pm"], "person": "John"},
      "original_text": "remind me to call John at 5 PM"
    }
  }'
```

---

## 🔐 Authentication Flow

### Firebase JWT Authentication

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Client Request │───▶│  Extract JWT    │───▶│  Validate Token │
│  with Auth      │    │  from Header    │    │  (Firebase)     │
│  Header         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Return User    │◀───│  Extract Claims │
                       │  Data (uid,     │    │  (uid, email,   │
                       │  email, name)   │    │  name)          │
                       └─────────────────┘    └─────────────────┘
```

**Authentication Methods:**
1. **Firebase Auth (Primary)**: JWT token validation
2. **Traditional Auth (Backup)**: Email/password with custom JWT
3. **Development Mode**: Relaxed validation for testing

**Headers Required:**
```
Authorization: Bearer <firebase_jwt_token>
Content-Type: application/json
```

---

## 🤖 AI Services Integration

### Speech-to-Text (STT) Service

**Supported Formats:**
- File Format: WAV
- Sample Rate: 16kHz
- Channels: Mono (1 channel)
- Bit Depth: 16-bit PCM
- Max File Size: 100MB

**Processing Flow:**
```
Audio File → Validation → Temporary Storage → STT Processing → Intent Classification → Cleanup → Response
```

**Key Endpoints:**
- `POST /stt/transcribe` - Convert audio to text only
- `POST /stt/transcribe-and-respond` - **ENHANCED: Complete voice-to-database pipeline**
- `GET /stt/model-info` - Service capabilities
- `GET /stt/voices` - Available TTS voices

#### Enhanced Voice-to-Database Pipeline

**`POST /stt/transcribe-and-respond`** - Complete pipeline that processes audio file through the entire workflow:

**Pipeline Steps:**
1. **Audio Validation** - Validates WAV format, sample rate, channels, bit depth
2. **Speech-to-Text** - Transcribes audio using Coqui STT
3. **Intent Classification** - Identifies user intent and extracts entities
4. **Database Processing** - Saves data to appropriate table based on intent
5. **AI Response** - Generates contextual response

**Request:**
```http
POST /api/v1/stt/transcribe-and-respond
Content-Type: multipart/form-data
Authorization: Bearer <firebase_token>

Form Data:
- audio_file: <WAV file> (required)
```

**Response:**
```json
{
  "success": true,
  "pipeline_completed": true,
  "processing_steps": {
    "audio_validation": true,
    "transcription": true,
    "intent_classification": true,
    "database_processing": true
  },
  "transcription": "add a reminder to call John at 5 PM",
  "intent_result": {
    "intent": "create_reminder",
    "confidence": 0.95,
    "entities": {
      "person": "John",
      "time": "5 PM"
    }
  },
  "processing_result": {
    "success": true,
    "message": "Reminder created successfully",
    "data": {
      "reminder_id": "uuid-here",
      "title": "Call John",
      "time": "2024-01-15T17:00:00Z",
      "person": "John"
    },
    "intent": "create_reminder"
  },
  "response_text": "I've created a reminder to call John at 5 PM.",
  "user_id": "user_id",
  "model_info": "coqui_stt_model_info",
  "audio_requirements": {
    "format": "WAV",
    "sample_rate": "16000Hz",
    "channels": "Mono",
    "bit_depth": "16-bit PCM"
  }
}
```

**Error Handling:**
- Validates each pipeline step independently
- Returns specific error messages for each step failure
- Automatic cleanup of temporary files
- Detailed logging for troubleshooting

**Database Integration:**
- `create_reminder` → `reminders` table
- `create_note` → `notes` table  
- `create_ledger` → `ledger_entries` table
- `chit_chat` → `history_logs` table

### Intent Classification Service

**Supported Intents:**
| Intent | Description | Entities | Example |
|--------|-------------|----------|---------|
| `create_reminder` | Create time-based reminders | time, person, date | "remind me to call John at 5 PM" |
| `create_note` | Create text notes | content | "note: meeting summary" |
| `create_ledger` | Track money/expenses | amount, person | "John owes me $50" |
| `add_expense` | Same as create_ledger | amount, person | "I spent $25 on lunch" |
| `chit_chat` | General conversation | none | "hello, how are you?" |
| `general_query` | Questions/queries | none | "what time is it?" |

**Entity Extraction Patterns:**
```javascript
Time Patterns:
- "5:00 PM", "3:30 AM" → Hour/minute with AM/PM
- "tomorrow", "today", "tonight" → Relative dates
- "in 2 hours", "in 30 minutes" → Relative time intervals

Person Patterns:
- "call John", "text Sarah" → Action + name
- "John owes me" → Name + financial context
- "meeting with Mike" → Context + name

Amount Patterns:
- "$50", "$25.50" → Currency symbols
- "25 dollars", "fifty bucks" → Text amounts
- "twenty-five" → Written numbers
```

### Text-to-Speech (TTS) Service

**Available Engines:**
- **pyttsx3**: Offline, Python 3.13 compatible
- **gTTS**: Online, Google Text-to-Speech

**Features:**
- Multiple voice options
- Language support (English, Spanish, French, German, etc.)
- Adjustable speech rate and volume
- Audio format: WAV/MP3

---

## 💾 Database Schema

### Core Tables

```sql
-- User Management
users (
    id VARCHAR PRIMARY KEY,           -- Firebase UID
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR,            -- For traditional auth
    language VARCHAR,
    timezone VARCHAR,
    created_at TIMESTAMP
);

preferences (
    user_id VARCHAR PRIMARY KEY,      -- FK to users.id
    allow_friends BOOLEAN,
    receive_shared_notes BOOLEAN,
    notification_sound VARCHAR,
    tts_language VARCHAR,
    chat_history_enabled BOOLEAN
);

-- Core Data Tables
reminders (
    id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,         -- FK to users.id
    title TEXT,
    description TEXT,
    time TIMESTAMP,
    repeat_pattern VARCHAR,
    timezone VARCHAR,
    is_shared BOOLEAN,
    created_by VARCHAR,               -- FK to users.id
    created_at TIMESTAMP
);

notes (
    id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,         -- FK to users.id
    content TEXT,
    source VARCHAR,                   -- 'voice_input', 'manual', etc.
    created_at TIMESTAMP
);

ledger_entries (
    id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,         -- FK to users.id
    contact_name VARCHAR,
    amount NUMERIC,
    direction VARCHAR CHECK (direction IN ('owe', 'owed')),
    created_at TIMESTAMP
);

-- Social Features
friendships (
    id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,         -- FK to users.id
    friend_id VARCHAR NOT NULL,       -- FK to users.id
    status VARCHAR CHECK (status IN ('pending', 'accepted', 'blocked')),
    created_at TIMESTAMP
);

-- System Tables
history_logs (
    id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,         -- FK to users.id
    content TEXT,
    interaction_type VARCHAR,         -- 'chit_chat', 'stt', 'command'
    created_at TIMESTAMP
);

embeddings (
    id UUID PRIMARY KEY,
    user_id VARCHAR NOT NULL,         -- FK to users.id
    reminder_id UUID,                 -- FK to reminders.id
    embedding FLOAT[],                -- Vector embeddings for search
    created_at TIMESTAMP
);
```

### Database Relationships

```
users (1) ←→ (1) preferences
users (1) ←→ (*) reminders
users (1) ←→ (*) notes
users (1) ←→ (*) ledger_entries
users (1) ←→ (*) friendships (as user)
users (1) ←→ (*) friendships (as friend)
users (1) ←→ (*) history_logs
users (1) ←→ (*) embeddings
reminders (1) ←→ (*) embeddings
```

---

## 🔧 Intent Processing Service

### Smart Entity Extraction

**Time Extraction:**
```python
# Examples of supported time formats:
"5:00 PM" → datetime(hour=17, minute=0)
"3:30 AM tomorrow" → datetime(next_day, hour=3, minute=30)
"in 2 hours" → datetime.now() + timedelta(hours=2)
"tomorrow morning" → datetime(next_day, hour=8, minute=0)
"tonight" → datetime(today, hour=20, minute=0)
```

**Person Extraction:**
```python
# Regex patterns for name extraction:
r'\b(?:call|contact|meet|see|tell|remind)\s+([A-Z][a-z]+)\b'
r'\b([A-Z][a-z]+)\s+(?:owes?|owed?)\b'
r'\b(?:to|about|with)\s+([A-Z][a-z]+)\b'

# Examples:
"call John" → person: "John"
"Sarah owes me" → person: "Sarah"
"meeting with Mike" → person: "Mike"
```

**Amount Extraction:**
```python
# Supported amount formats:
"$50" → 50.0
"25 dollars" → 25.0
"fifty bucks" → 50.0
"$125.50" → 125.50

# Direction detection:
"John owes me $50" → direction: "owed" (they owe me)
"I owe Sarah $25" → direction: "owe" (I owe them)
```

### Title Generation

**Smart Title Creation:**
```python
# Examples of auto-generated titles:
"remind me to call John at 5 PM" → "Call John"
"don't forget to buy groceries" → "Buy groceries"
"meeting with Sarah tomorrow" → "Meeting with Sarah"
"note about project ideas" → "Note about project"
```

---

## 📝 API Request/Response Examples

### 1. Voice Transcription

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/stt/transcribe" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -F "audio_file=@audio.wav"
```

**Response:**
```json
{
  "success": true,
  "transcription": "remind me to call John at 5 PM",
  "intent": {
    "intent": "create_reminder",
    "confidence": 0.95,
    "entities": {
      "time": ["5", "00", "pm"],
      "person": "John"
    },
    "original_text": "remind me to call John at 5 PM"
  },
  "user_id": "firebase_uid_123",
  "model_info": {
    "model_type": "coqui_stt",
    "status": "ready"
  }
}
```

### 2. Intent Processing

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/intent-processor/process" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "intent_data": {
      "intent": "create_reminder",
      "confidence": 0.95,
      "entities": {"time": ["5", "00", "pm"], "person": "John"},
      "original_text": "remind me to call John at 5 PM"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Reminder created successfully",
  "data": {
    "reminder_id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "Call John",
    "description": "remind me to call John at 5 PM",
    "time": "2025-05-29T17:00:00",
    "person": "John"
  },
  "intent": "create_reminder"
}
```

### 3. Ledger Entry Creation

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/intent-processor/process" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "intent_data": {
      "intent": "create_ledger",
      "confidence": 0.88,
      "entities": {"amount": 50.0, "person": "Sarah"},
      "original_text": "Sarah owes me $50 for dinner"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Ledger entry created successfully",
  "data": {
    "ledger_id": "456e7890-e89b-12d3-a456-426614174001",
    "contact_name": "Sarah",
    "amount": 50.0,
    "direction": "owed",
    "description": "Sarah owed $50.0"
  },
  "intent": "create_ledger"
}
```

### 4. CRUD Operations

**Get User's Reminders:**
```bash
curl -X GET "http://localhost:8000/api/v1/reminders/" \
  -H "Authorization: Bearer $FIREBASE_JWT"
```

**Update Reminder:**
```bash
curl -X PUT "http://localhost:8000/api/v1/reminders/{reminder_id}" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Updated title",
    "time": "2025-05-30T18:00:00"
  }'
```

**Delete Reminder:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/reminders/{reminder_id}" \
  -H "Authorization: Bearer $FIREBASE_JWT"
```

---

## ⚙️ Configuration & Environment

### Required Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:admin123@localhost:5433/eindr

# Authentication
DEV_MODE=true                    # Enable development mode
SECRET_KEY=your-secret-key-here

# Server Configuration
HOST=127.0.0.1
PORT=8000
DEBUG=true

# AI Model Paths
COQUI_STT_MODEL_PATH=./models/coqui-stt.tflite
TTS_MODEL_PATH=./models/tts
INTENT_MODEL_PATH=./models/intent

# File Upload
MAX_FILE_SIZE=104857600          # 100MB
UPLOAD_DIR=./uploads
```

### Service Dependencies

```yaml
Services Required:
  ✅ PostgreSQL Database (port 5433)
  ✅ Firebase Authentication
  ✅ Speech-to-Text Service (Coqui STT)
  ✅ Text-to-Speech Service (pyttsx3/gTTS)
  ✅ Intent Classification Service
  ✅ Chat/Conversation Service
  ✅ File Upload System
  ✅ Background Task Scheduler

Python Dependencies:
  ✅ FastAPI + Uvicorn
  ✅ SQLAlchemy + PostgreSQL driver
  ✅ Pydantic for data validation
  ✅ Firebase Auth libraries
  ✅ AI/ML libraries (compatible with Python 3.13)
```

---

## 🚀 Deployment & Usage

### 1. Application Startup

```bash
# Start the application
python main.py

# Or with uvicorn directly
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Health Check

```bash
# Check overall health
curl http://localhost:8000/health

# Check specific endpoints
curl http://localhost:8000/api/v1/intent-processor/supported-intents
```

### 3. Database Initialization

```bash
# Initialize database tables
python init_db.py

# Or use the simple table creation script
python simple_create_tables.py
```

---

## 🔍 Error Handling & Validation

### Authentication Errors
- **401 Unauthorized**: Missing or invalid JWT token
- **403 Forbidden**: Valid token but insufficient permissions

### Validation Errors
- **400 Bad Request**: Invalid request format or missing required fields
- **422 Unprocessable Entity**: Data validation failed

### Service Errors
- **503 Service Unavailable**: AI services not ready or loaded
- **500 Internal Server Error**: Database or processing errors

### File Upload Errors
- **400 Bad Request**: Unsupported file format or size too large
- **415 Unsupported Media Type**: Invalid audio format

---

## 📊 Performance & Monitoring

### Database Optimization
- Connection pooling with SQLAlchemy
- Proper indexing on foreign keys
- Cascading deletes for data integrity

### Service Monitoring
- Comprehensive logging with structured format
- Health check endpoints for all services
- Background task scheduling and monitoring

### Caching Strategy
- Service-level caching for AI models
- Session management for database connections
- Temporary file cleanup for uploads

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Advanced NLP with transformer models
- [ ] Multi-language support expansion
- [ ] Real-time notifications via WebSocket
- [ ] Enhanced vector search capabilities
- [ ] Advanced friend sharing permissions
- [ ] Recurring reminder patterns
- [ ] Voice-based conversation flows
- [ ] Integration with external calendar systems

### Scalability Considerations
- [ ] Microservices architecture migration
- [ ] Redis caching layer
- [ ] Message queue for background processing
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Load balancing for AI services

---

**🎉 End of Documentation**

*This documentation provides a comprehensive overview of the Eindr Backend API flow, covering all major components, endpoints, and integration patterns. The system successfully implements a complete voice-to-database pipeline with intelligent AI processing and robust data management.* 