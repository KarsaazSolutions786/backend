# 🤖 Eindr Backend - Complete Speech & Intent Processing Workflow Documentation

## 📋 Overview
**Eindr** is an AI-powered reminder application backend that processes voice inputs through a complete pipeline: Speech-to-Text → Intent Classification → Database Storage → AI Response. This documentation provides the complete workflow for the **Speech (`/api/v1/stt/*`)** and **Intent Processing (`/api/v1/intent-processor/*`)** systems.

---

## 🏗️ System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Audio Input       │    │   Speech Pipeline   │    │   Database Storage  │
│   ├── WAV Files     │───▶│   ├── STT Service   │───▶│   ├── Reminders     │
│   ├── 16kHz Mono    │    │   ├── Intent AI     │    │   ├── Notes         │
│   └── 16-bit PCM    │    │   └── DB Integration│    │   └── Ledger        │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │
                           ┌─────────────────────┐
                           │   Firebase Auth     │
                           │   └── JWT Tokens    │
                           └─────────────────────┘
```

---

## 🎯 Speech-to-Text API (`/api/v1/stt/*`)

### **Core Endpoints**

#### 1. **Complete Voice-to-Database Pipeline**
```http
POST /api/v1/stt/transcribe-and-respond
```
**Primary endpoint for full voice processing workflow**

**Request:**
- **Content-Type**: `multipart/form-data`
- **Authorization**: `Bearer <firebase_jwt_token>`
- **Form Data**: `audio_file` (WAV file)

**Audio Requirements:**
- **Format**: WAV (.wav)
- **Sample Rate**: 16 kHz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit PCM
- **Max Size**: 100MB

**Pipeline Steps:**
1. **Audio Validation** → Validates WAV format and specifications
2. **Speech-to-Text** → Transcribes audio using Coqui STT
3. **Intent Classification** → Identifies user intent and extracts entities
4. **Database Processing** → Saves data to appropriate table
5. **AI Response** → Generates contextual response

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
  "model_info": "coqui_stt_model_info"
}
```

#### 2. **Transcription Only**
```http
POST /api/v1/stt/transcribe
```
**Converts audio to text without intent processing**

**Response:**
```json
{
  "success": true,
  "transcription": "add a reminder to call John at 5 PM",
  "confidence": 0.95,
  "user_id": "user_id",
  "model_info": "coqui_stt_model_info"
}
```

#### 3. **Intent Classification**
```http
POST /api/v1/stt/intent-classify
```
**Classifies intent from text input**

**Request Body:**
```json
{
  "text": "add a reminder to call John at 5 PM"
}
```

**Response:**
```json
{
  "success": true,
  "intent": "create_reminder",
  "confidence": 0.95,
  "entities": {
    "person": "John",
    "time": "5 PM"
  },
  "original_text": "add a reminder to call John at 5 PM"
}
```

#### 4. **Text-to-Speech Response**
```http
GET /api/v1/stt/response-audio/{text}?voice=default
```
**Generates audio response from text**

**Parameters:**
- `text`: Text to convert to speech
- `voice`: Voice selection (optional, default: "default")

**Response:** Audio stream (WAV format)

#### 5. **Service Information**
```http
GET /api/v1/stt/model-info
GET /api/v1/stt/voices
```
**Get STT model capabilities and available TTS voices**

---

## 🧠 Intent Processing API (`/api/v1/intent-processor/*`)

### **Advanced Intent Classification System**

#### 1. **Comprehensive Intent Classification**
```http
POST /api/v1/intent-processor/classify
```
**Advanced intent classification with multi-intent support**

**Request:**
```json
{
  "text": "remind me to call John at 5 PM and create a note about the meeting",
  "multi_intent": true,
  "include_entities": true,
  "confidence_threshold": 0.1
}
```

**Response:**
```json
{
  "success": true,
  "type": "multi_intent",
  "intents": ["create_reminder", "create_note"],
  "overall_confidence": 0.92,
  "segments": [
    "remind me to call John at 5 PM",
    "create a note about the meeting"
  ],
  "results": [
    {
      "intent": "create_reminder",
      "confidence": 0.95,
      "entities": {"person": "John", "time": "5 PM"},
      "segment": "remind me to call John at 5 PM"
    },
    {
      "intent": "create_note",
      "confidence": 0.89,
      "entities": {"content": "meeting"},
      "segment": "create a note about the meeting"
    }
  ],
  "original_text": "remind me to call John at 5 PM and create a note about the meeting",
  "model_used": "pytorch"
}
```

#### 2. **Classify and Store in Database**
```http
POST /api/v1/intent-processor/classify-and-store
```
**Complete pipeline: Classification + Database Storage**

**Request:**
```json
{
  "text": "remind me to call John at 5 PM",
  "multi_intent": true,
  "include_entities": true,
  "store_in_database": true
}
```

**Response:**
```json
{
  "success": true,
  "classification_result": {
    "intent": "create_reminder",
    "confidence": 0.95,
    "entities": {"person": "John", "time": "5 PM"}
  },
  "storage_result": {
    "success": true,
    "reminder_id": "uuid-here",
    "table": "reminders",
    "message": "Reminder created successfully"
  },
  "user_id": "user_id"
}
```

#### 3. **Batch Processing**
```http
POST /api/v1/intent-processor/batch-classify
```
**Process multiple texts simultaneously**

**Request:**
```json
{
  "texts": [
    "remind me to call John",
    "note about the meeting",
    "Sarah owes me $50"
  ],
  "multi_intent": true,
  "store_results": true
}
```

#### 4. **Database Operations**
```http
POST /api/v1/intent-processor/process-and-store
GET /api/v1/intent-processor/user-statistics
GET /api/v1/intent-processor/recent-records/{table_name}
```

---

## 🎯 Supported Intents & Database Mapping

### **Intent Types**

| Intent | Description | Database Table | Entities | Example |
|--------|-------------|----------------|----------|---------|
| **create_reminder** | Time-based reminders | `reminders` | time, person, date, task | "remind me to call John at 5 PM" |
| **create_note** | Text notes & memos | `notes` | content, topic | "note: meeting summary about project" |
| **create_ledger** | Money/debt tracking | `ledger_entries` | amount, person, direction | "John owes me $50" |
| **add_expense** | Expense tracking | `ledger_entries` | amount, category, description | "I spent $25 on lunch" |
| **chit_chat** | General conversation | `history_logs` | message_type | "hello, how are you?" |
| **general_query** | Questions/requests | `history_logs` | query_type | "what's the weather like?" |

### **Entity Extraction**

#### **Time Entities:**
- **Formats**: "5 PM", "tomorrow", "in 2 hours", "next Friday"
- **Processing**: Converts to ISO datetime format
- **Examples**: 
  - "5 PM" → "2024-01-15T17:00:00Z"
  - "tomorrow" → "2024-01-16T09:00:00Z"

#### **Person Entities:**
- **Detection**: Names, pronouns, relationships
- **Examples**: "John", "mom", "my boss", "Sarah"

#### **Amount Entities:**
- **Formats**: "$50", "fifty dollars", "25.99"
- **Processing**: Extracts numeric value and currency
- **Direction**: Determines who owes whom

---

## 🔄 Complete Workflow Examples

### **Example 1: Voice Reminder Creation**

**Step 1: Audio Upload**
```bash
curl -X POST "http://localhost:8000/api/v1/stt/transcribe-and-respond" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -F "audio_file=@reminder_call_john.wav"
```

**Audio Content:** *"Add a reminder to call John at 5 PM"*

**Step 2: Automatic Processing**
1. **Audio Validation** ✅ WAV, 16kHz, Mono, 16-bit
2. **STT Transcription** ✅ "add a reminder to call John at 5 PM"
3. **Intent Classification** ✅ Intent: "create_reminder", Confidence: 0.95
4. **Entity Extraction** ✅ person: "John", time: "5 PM"
5. **Database Storage** ✅ New record in `reminders` table
6. **AI Response** ✅ "I've created a reminder to call John at 5 PM."

**Final Response:**
```json
{
  "success": true,
  "pipeline_completed": true,
  "transcription": "add a reminder to call John at 5 PM",
  "intent_result": {
    "intent": "create_reminder",
    "confidence": 0.95,
    "entities": {"person": "John", "time": "5 PM"}
  },
  "processing_result": {
    "success": true,
    "data": {
      "reminder_id": "abc-123",
      "title": "Call John",
      "scheduled_time": "2024-01-15T17:00:00Z",
      "contact_name": "John"
    }
  },
  "response_text": "I've created a reminder to call John at 5 PM."
}
```

### **Example 2: Multi-Intent Processing**

**Input:** *"Remind me to call Sarah tomorrow and note that she owes me $30"*

**Processing:**
1. **Segmentation**: 
   - Segment 1: "remind me to call Sarah tomorrow"
   - Segment 2: "note that she owes me $30"

2. **Intent Classification**:
   - Intent 1: "create_reminder" (confidence: 0.94)
   - Intent 2: "create_ledger" (confidence: 0.91)

3. **Database Operations**:
   - Creates reminder in `reminders` table
   - Creates ledger entry in `ledger_entries` table

**Response:**
```json
{
  "success": true,
  "type": "multi_intent",
  "results": [
    {
      "intent": "create_reminder",
      "success": true,
      "data": {"reminder_id": "def-456", "title": "Call Sarah", "time": "tomorrow"}
    },
    {
      "intent": "create_ledger", 
      "success": true,
      "data": {"ledger_id": "ghi-789", "amount": 30.0, "contact_name": "Sarah", "direction": "owed"}
    }
  ],
  "response_text": "Perfect! I've created a reminder to call Sarah tomorrow and recorded that she owes you $30."
}
```

---

## 🔐 Authentication & Security

### **Firebase JWT Authentication**
All endpoints require Firebase authentication:

```http
Authorization: Bearer <firebase_jwt_token>
Content-Type: application/json (for JSON) | multipart/form-data (for files)
```

### **User Context**
- User ID extracted from JWT token
- All database records linked to authenticated user
- Proper authorization validation on each request

---

## ⚠️ Error Handling & Edge Cases

### **Audio Processing Errors**
```json
{
  "success": false,
  "error": "Invalid audio format",
  "requirements": {
    "format": "WAV",
    "sample_rate": "16000Hz",
    "channels": "Mono",
    "bit_depth": "16-bit PCM"
  },
  "received": {
    "format": "MP3",
    "sample_rate": "44100Hz"
  }
}
```

### **Intent Classification Failures**
```json
{
  "success": false,
  "error": "Intent classification failed",
  "fallback_intent": "chit_chat",
  "confidence": 0.1,
  "original_text": "unclear speech result"
}
```

### **Database Processing Errors**
```json
{
  "success": false,
  "error": "Database processing failed",
  "details": "Invalid time format",
  "intent": "create_reminder",
  "entities": {"time": "invalid_time_string"}
}
```

---

## 🚀 Usage Recommendations

### **For Optimal Results**
1. **Audio Quality**: Use clear, noise-free audio recordings
2. **Speech Clarity**: Speak clearly with natural pacing
3. **Specific Requests**: Include specific details (times, names, amounts)
4. **Supported Phrases**: Use natural language matching supported intents

### **Best Practices**
- Test audio requirements before production use
- Implement client-side audio validation
- Handle partial failures gracefully
- Provide user feedback during processing
- Implement retry logic for failed requests

---

## 📚 Quick Reference

### **Common cURL Examples**

#### Voice-to-Database Pipeline
```bash
curl -X POST "http://localhost:8000/api/v1/stt/transcribe-and-respond" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -F "audio_file=@audio.wav"
```

#### Text Intent Classification
```bash
curl -X POST "http://localhost:8000/api/v1/intent-processor/classify" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "remind me to call John at 5 PM",
    "multi_intent": true,
    "include_entities": true
  }'
```

#### Batch Processing
```bash
curl -X POST "http://localhost:8000/api/v1/intent-processor/batch-classify" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["remind me to call John", "note about meeting"],
    "store_results": true
  }'
```

---

## 🔧 Development Setup

### **Environment Variables**
```env
# Firebase Configuration
FIREBASE_SERVICE_ACCOUNT_PATH=firebase-service-account.json

# STT/TTS Models
VOSK_MODEL_PATH=./models/vosk-model-en-us-0.22
TTS_MODEL_PATH=./models/tts
INTENT_MODEL_PATH=./models/intent

# Database
DATABASE_URL=postgresql://postgres:admin123@localhost:5432/eindr
```

### **Testing Audio Files**
Create test WAV files with these specifications:
- 16 kHz sample rate
- Mono channel
- 16-bit PCM encoding
- Clear speech content

### **Model Requirements**
- **Vosk Model**: Download English STT model
- **Coqui TTS**: Text-to-speech model files
- **Intent Model**: PyTorch-based classification model

---

This documentation provides complete understanding of how to interact with the Eindr speech and intent processing systems, including all available endpoints, expected request/response formats, and the complete voice-to-database workflow. 