# Eindr Microservices Backend - Complete Project Overview

## 🏗️ Architecture Overview

**Eindr** is an enterprise-grade AI-powered personal assistant platform built with a comprehensive microservices architecture. The system consists of 14 microservices orchestrated through Docker Compose with Kong API Gateway as the entry point.

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │────│  Kong Gateway    │────│  Microservices  │
│   Applications  │    │  (Port 8080)     │    │  (14 Services)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ PostgreSQL  │ │   Redis     │ │  RabbitMQ   │ │Prometheus │ │
│  │ (Port 5433) │ │(Port 6379)  │ │(Port 5672)  │ │(Port 9090)│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Services Overview

### Core Infrastructure Services

| Service | Port | Purpose | Health Check |
|---------|------|---------|-------------|
| Kong Gateway | 8080 | API Gateway & Routing | http://localhost:8080/health |
| PostgreSQL | 5433 | Primary Database | Internal |
| Redis | 6379 | Caching & Sessions | Internal |
| RabbitMQ | 5672 | Message Queue | http://localhost:15672 |
| Prometheus | 9090 | Metrics Collection | http://localhost:9090 |
| Grafana | 3001 | Monitoring Dashboard | http://localhost:3001 |

### Business Logic Microservices

| Service | Port | Purpose | API Prefix |
|---------|------|---------|------------|
| Auth Service | 8001 | Authentication & Authorization | `/auth` |
| Customer Service | 8002 | Customer Profile Management | `/customers` |
| Reminder Service | 8003 | Task & Reminder Management | `/reminders` |
| Note Service | 8004 | Note-taking & Organization | `/notes` |
| Ledger Service | 8005 | Financial Tracking | `/ledger` |
| Friend Service | 8006 | Social Connections | `/friends` |
| History Service | 8007 | Activity Tracking | `/history` |
| Chat Service | 8011 | Conversation Management | `/chat` |
| Scheduler Service | 8012 | Task Scheduling | `/scheduler` |

### AI Services

| Service | Port | Purpose | Technology |
|---------|------|---------|------------|
| STT Service | 8008 | Speech-to-Text | Whisper |
| TTS Service | 8009 | Text-to-Speech | Coqui TTS |
| Intent Service | 8010 | Intent Classification | MiniLM |
| AI Pipeline Service | 8081 | AI Workflow Orchestration | Custom |

## 🔐 Authentication Flow

### JWT Token-Based Authentication

All services use JWT tokens for authentication. The auth service issues tokens that other services validate.

```mermaid
sequenceDiagram
    participant Client
    participant Kong
    participant Auth
    participant Service
    
    Client->>Kong: POST /auth/login
    Kong->>Auth: Forward request
    Auth->>Auth: Validate credentials
    Auth->>Client: Return JWT tokens
    
    Client->>Kong: API request + JWT
    Kong->>Service: Forward with JWT
    Service->>Service: Validate JWT
    Service->>Client: Return response
```

## 📚 API Documentation with Examples

### 1. Authentication Service (Port 8001)

#### Register New Customer

**Endpoint:** `POST /auth/register`

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "confirm_password": "SecurePass123!",
  "full_name": "John Doe",
  "gender": "male",
  "is_new": true
}
```

**Response (201 Created):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 900,
  "customer": {
    "id": 1,
    "email": "user@example.com",
    "is_verified": false,
    "is_active": true,
    "created_at": "2024-01-15T10:30:00Z",
    "last_login": null,
    "login_attempts": 0,
    "locked_until": null,
    "subscription_plan_id": null,
    "profile": {
      "full_name": "John Doe",
      "gender": "male",
      "is_new": true
    }
  }
}
```

#### Login Customer

**Endpoint:** `POST /auth/login`

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "remember_me": false
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 900,
  "customer": {
    "id": 1,
    "email": "user@example.com",
    "is_verified": true,
    "is_active": true,
    "created_at": "2024-01-15T10:30:00Z",
    "last_login": "2024-01-15T14:45:00Z",
    "login_attempts": 0,
    "locked_until": null,
    "subscription_plan_id": 1,
    "profile": {
      "full_name": "John Doe",
      "gender": "male",
      "is_new": false
    }
  }
}
```

#### Refresh Token

**Endpoint:** `POST /auth/refresh`

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 900
}
```

### 2. Customer Service (Port 8002)

#### Get Customer Profile

**Endpoint:** `GET /customers/profile`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "profile": {
    "full_name": "John Doe",
    "user_name": "johndoe",
    "bio": "AI enthusiast and productivity lover",
    "avatar_url": "https://example.com/avatar.jpg",
    "phone_number": "+1234567890",
    "date_of_birth": "1990-05-15",
    "timezone_id": 1,
    "language_id": 1,
    "country": "United States",
    "city": "San Francisco",
    "gender": "male",
    "is_new": false
  },
  "subscription_plan": {
    "id": 1,
    "plan_name": "Premium",
    "price": 29.99,
    "billing_interval": "monthly",
    "max_seats": 5
  }
}
```

#### Update Customer Profile

**Endpoint:** `PUT /customers/profile`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "full_name": "John Smith",
  "bio": "Updated bio - Tech entrepreneur",
  "phone_number": "+1987654321",
  "country": "Canada",
  "city": "Toronto"
}
```

**Response (200 OK):**
```json
{
  "message": "Profile updated successfully",
  "profile": {
    "full_name": "John Smith",
    "bio": "Updated bio - Tech entrepreneur",
    "phone_number": "+1987654321",
    "country": "Canada",
    "city": "Toronto",
    "updated_at": "2024-01-15T15:30:00Z"
  }
}
```

### 3. Reminder Service (Port 8003)

#### Create Reminder

**Endpoint:** `POST /reminders`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "title": "Team Meeting",
  "description": "Weekly team sync meeting",
  "reminder_time": "2024-01-16T09:00:00Z",
  "priority": "high",
  "category": "work",
  "is_recurring": true,
  "recurrence_pattern": "weekly",
  "tags": ["meeting", "team", "work"]
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "customer_id": 1,
  "title": "Team Meeting",
  "description": "Weekly team sync meeting",
  "reminder_time": "2024-01-16T09:00:00Z",
  "priority": "high",
  "category": "work",
  "is_recurring": true,
  "recurrence_pattern": "weekly",
  "status": "active",
  "tags": ["meeting", "team", "work"],
  "created_at": "2024-01-15T15:45:00Z",
  "updated_at": "2024-01-15T15:45:00Z"
}
```

#### Get Reminders

**Endpoint:** `GET /reminders?status=active&limit=10&offset=0`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "reminders": [
    {
      "id": 1,
      "title": "Team Meeting",
      "description": "Weekly team sync meeting",
      "reminder_time": "2024-01-16T09:00:00Z",
      "priority": "high",
      "category": "work",
      "status": "active",
      "tags": ["meeting", "team", "work"]
    },
    {
      "id": 2,
      "title": "Doctor Appointment",
      "description": "Annual checkup",
      "reminder_time": "2024-01-17T14:30:00Z",
      "priority": "medium",
      "category": "health",
      "status": "active",
      "tags": ["health", "appointment"]
    }
  ],
  "total": 2,
  "limit": 10,
  "offset": 0
}
```

### 4. Note Service (Port 8004)

#### Create Note

**Endpoint:** `POST /notes`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "title": "Project Ideas",
  "content": "# AI Assistant Features\n\n- Voice commands\n- Smart scheduling\n- Context awareness",
  "category": "work",
  "tags": ["ai", "project", "ideas"],
  "is_favorite": false,
  "is_archived": false
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "customer_id": 1,
  "title": "Project Ideas",
  "content": "# AI Assistant Features\n\n- Voice commands\n- Smart scheduling\n- Context awareness",
  "category": "work",
  "tags": ["ai", "project", "ideas"],
  "is_favorite": false,
  "is_archived": false,
  "created_at": "2024-01-15T16:00:00Z",
  "updated_at": "2024-01-15T16:00:00Z"
}
```

#### Search Notes

**Endpoint:** `GET /notes/search?q=AI&category=work&limit=5`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "notes": [
    {
      "id": 1,
      "title": "Project Ideas",
      "content": "# AI Assistant Features\n\n- Voice commands...",
      "category": "work",
      "tags": ["ai", "project", "ideas"],
      "is_favorite": false,
      "created_at": "2024-01-15T16:00:00Z",
      "relevance_score": 0.95
    }
  ],
  "total": 1,
  "query": "AI",
  "filters": {
    "category": "work"
  }
}
```

### 5. Ledger Service (Port 8005)

#### Create Ledger Entry

**Endpoint:** `POST /ledger/entries`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "type": "expense",
  "amount": 45.99,
  "currency": "USD",
  "description": "Lunch at restaurant",
  "category": "food",
  "date": "2024-01-15",
  "tags": ["restaurant", "lunch"],
  "location": "Downtown Cafe",
  "payment_method": "credit_card"
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "customer_id": 1,
  "type": "expense",
  "amount": 45.99,
  "currency": "USD",
  "description": "Lunch at restaurant",
  "category": "food",
  "date": "2024-01-15",
  "tags": ["restaurant", "lunch"],
  "location": "Downtown Cafe",
  "payment_method": "credit_card",
  "created_at": "2024-01-15T16:15:00Z",
  "updated_at": "2024-01-15T16:15:00Z"
}
```

#### Get Financial Summary

**Endpoint:** `GET /ledger/summary?period=month&year=2024&month=1`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (200 OK):**
```json
{
  "period": "2024-01",
  "summary": {
    "total_income": 3500.00,
    "total_expenses": 1245.67,
    "net_income": 2254.33,
    "currency": "USD"
  },
  "categories": {
    "food": {
      "total": 345.99,
      "count": 12,
      "percentage": 27.8
    },
    "transport": {
      "total": 189.50,
      "count": 8,
      "percentage": 15.2
    },
    "entertainment": {
      "total": 156.78,
      "count": 5,
      "percentage": 12.6
    }
  },
  "trends": {
    "vs_last_month": {
      "income_change": "+5.2%",
      "expense_change": "-2.1%"
    }
  }
}
```

### 6. Chat Service (Port 8011)

#### Create Conversation

**Endpoint:** `POST /chat/conversations`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "title": "Planning Weekend Trip",
  "context": "travel",
  "initial_message": "Help me plan a weekend trip to San Francisco"
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "customer_id": 1,
  "title": "Planning Weekend Trip",
  "context": "travel",
  "status": "active",
  "created_at": "2024-01-15T16:30:00Z",
  "updated_at": "2024-01-15T16:30:00Z",
  "message_count": 1,
  "last_message": {
    "id": 1,
    "role": "user",
    "content": "Help me plan a weekend trip to San Francisco",
    "timestamp": "2024-01-15T16:30:00Z"
  }
}
```

#### Send Message

**Endpoint:** `POST /chat/conversations/1/messages`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "content": "What are the best attractions to visit?",
  "message_type": "text"
}
```

**Response (201 Created):**
```json
{
  "id": 2,
  "conversation_id": 1,
  "role": "user",
  "content": "What are the best attractions to visit?",
  "message_type": "text",
  "timestamp": "2024-01-15T16:32:00Z",
  "ai_response": {
    "id": 3,
    "role": "assistant",
    "content": "Here are some top attractions in San Francisco:\n\n1. **Golden Gate Bridge** - Iconic landmark with stunning views\n2. **Alcatraz Island** - Historic prison tour\n3. **Fisherman's Wharf** - Waterfront dining and shopping\n4. **Lombard Street** - The world's crookedest street\n5. **Chinatown** - Largest outside of Asia\n\nWould you like specific recommendations based on your interests?",
    "timestamp": "2024-01-15T16:32:15Z",
    "confidence_score": 0.92
  }
}
```

### 7. AI Pipeline Service (Port 8081)

#### Process Audio Pipeline

**Endpoint:** `POST /pipeline/audio-pipeline`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: multipart/form-data
```

**Request Body (Form Data):**
```
audio_file: [audio file binary]
language: "en"
context: "general"
```

**Response (200 OK):**
```json
{
  "pipeline_id": "pip_1234567890",
  "status": "completed",
  "processing_time": 2.45,
  "results": {
    "stt": {
      "transcription": "Set a reminder for my meeting tomorrow at 2 PM",
      "confidence": 0.94,
      "language": "en",
      "processing_time": 0.8
    },
    "intent": {
      "intent": "create_reminder",
      "confidence": 0.89,
      "entities": {
        "time": "tomorrow at 2 PM",
        "event": "meeting"
      },
      "processing_time": 0.3
    },
    "chat": {
      "response": "I'll help you set a reminder for your meeting tomorrow at 2 PM. Let me create that for you.",
      "action": "create_reminder",
      "parameters": {
        "title": "Meeting",
        "time": "2024-01-16T14:00:00Z",
        "priority": "medium"
      },
      "processing_time": 0.9
    },
    "tts": {
      "audio_url": "https://storage.example.com/audio/response_1234567890.wav",
      "duration": 3.2,
      "voice": "female_en",
      "processing_time": 0.45
    }
  },
  "actions_performed": [
    {
      "service": "reminder-service",
      "action": "create_reminder",
      "status": "success",
      "reminder_id": 15
    }
  ]
}
```

#### Text-Only Pipeline

**Endpoint:** `POST /pipeline/text-pipeline`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "text": "Show me my expenses for this month",
  "context": "financial",
  "include_tts": true
}
```

**Response (200 OK):**
```json
{
  "pipeline_id": "pip_1234567891",
  "status": "completed",
  "processing_time": 1.2,
  "results": {
    "intent": {
      "intent": "get_expenses",
      "confidence": 0.96,
      "entities": {
        "period": "this month",
        "type": "expenses"
      }
    },
    "chat": {
      "response": "Here are your expenses for this month: Total: $1,245.67. Top categories: Food ($345.99), Transport ($189.50), Entertainment ($156.78). You're 12% under budget!",
      "data": {
        "total_expenses": 1245.67,
        "categories": [
          {"name": "food", "amount": 345.99},
          {"name": "transport", "amount": 189.50},
          {"name": "entertainment", "amount": 156.78}
        ]
      }
    },
    "tts": {
      "audio_url": "https://storage.example.com/audio/response_1234567891.wav",
      "duration": 8.5
    }
  }
}
```

### 8. STT Service (Port 8008)

#### Transcribe Audio

**Endpoint:** `POST /stt/transcribe`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: multipart/form-data
```

**Request Body (Form Data):**
```
audio_file: [audio file binary]
language: "en"
model: "whisper-base"
```

**Response (200 OK):**
```json
{
  "transcription": "Hello, can you help me schedule a meeting for next Tuesday?",
  "confidence": 0.94,
  "language": "en",
  "duration": 3.2,
  "processing_time": 0.8,
  "segments": [
    {
      "start": 0.0,
      "end": 1.2,
      "text": "Hello, can you help me",
      "confidence": 0.96
    },
    {
      "start": 1.2,
      "end": 3.2,
      "text": "schedule a meeting for next Tuesday?",
      "confidence": 0.92
    }
  ]
}
```

### 9. TTS Service (Port 8009)

#### Synthesize Speech

**Endpoint:** `POST /tts/synthesize`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "text": "Your meeting has been scheduled for next Tuesday at 2 PM.",
  "voice": "female_en",
  "speed": 1.0,
  "format": "wav"
}
```

**Response (200 OK):**
```json
{
  "audio_url": "https://storage.example.com/audio/tts_1234567892.wav",
  "duration": 4.1,
  "format": "wav",
  "voice": "female_en",
  "processing_time": 0.6,
  "text_length": 58,
  "sample_rate": 22050
}
```

### 10. Intent Service (Port 8010)

#### Classify Intent

**Endpoint:** `POST /intent/classify`

**Headers:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request Body:**
```json
{
  "text": "Can you remind me to call mom tomorrow?",
  "context": "general"
}
```

**Response (200 OK):**
```json
{
  "intent": "create_reminder",
  "confidence": 0.92,
  "entities": {
    "action": "call",
    "person": "mom",
    "time": "tomorrow"
  },
  "alternative_intents": [
    {
      "intent": "create_task",
      "confidence": 0.15
    },
    {
      "intent": "schedule_event",
      "confidence": 0.08
    }
  ],
  "processing_time": 0.2
}
```

## 🔄 Inter-Service Communication

### Message Queue (RabbitMQ)

Services communicate asynchronously through RabbitMQ for:

- **Event notifications** (user actions, system events)
- **Background processing** (email sending, data sync)
- **Service coordination** (workflow orchestration)

### Example Message Flow

```json
{
  "event_type": "reminder_created",
  "timestamp": "2024-01-15T16:45:00Z",
  "customer_id": 1,
  "data": {
    "reminder_id": 15,
    "title": "Team Meeting",
    "reminder_time": "2024-01-16T09:00:00Z"
  },
  "source_service": "reminder-service",
  "correlation_id": "req_1234567890"
}
```

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM
- 20GB+ disk space

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd microservices/backend

# Start all services
make up

# Check service health
make health-check

# View logs
make logs
```

### Service URLs

- **API Gateway**: http://localhost:8080
- **Kong Admin**: http://localhost:8101
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **RabbitMQ**: http://localhost:15672 (eindr/eindr123)
- **PgAdmin**: http://localhost:5050 (admin@admin.com/admin)

### Development Commands

```bash
# Build services
make build

# Run migrations
make migrate-all

# Setup Kong gateway
make kong-setup

# Reset environment
make reset

# View specific service logs
docker-compose -f docker-compose.microservices.yml logs -f auth-service
```

## 🔧 Configuration

### Environment Variables

Key environment variables for each service:

```bash
# Database
DATABASE_URL=postgresql://eindr:eindr_pass@new-postgres-server:5432/eindr_db

# Redis
REDIS_URL=redis://redis:6379

# RabbitMQ
RABBITMQ_URL=amqp://eindr:eindr123@rabbitmq:5672/

# JWT
SECRET_KEY=eindr-super-secure-jwt-secret-key-for-production-2024-v1

# Service URLs
AUTH_SERVICE_URL=http://auth-service:8000
```

### Security Configuration

- **CORS**: Configured per environment
- **Rate Limiting**: 100 requests/minute default
- **JWT Expiration**: 15 minutes (access), 7 days (refresh)
- **Password Policy**: Minimum 8 characters
- **Account Lockout**: 5 failed attempts, 30-minute lockout

## 📊 Monitoring & Observability

### Metrics Collection

- **Prometheus** scrapes metrics from all services
- **Grafana** provides visualization dashboards
- **Health checks** on all endpoints
- **Request tracing** with correlation IDs

### Key Metrics

- Request rate and latency
- Error rates by service
- Database connection pools
- AI model inference times
- Queue message processing

## 🧪 Testing

### API Testing

Use the provided Postman collection or curl commands:

```bash
# Test authentication
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Test with JWT token
curl -X GET http://localhost:8080/customers/profile \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Load Testing

```bash
# Install artillery
npm install -g artillery

# Run load test
artillery run tests/load-test.yml
```

## 🔒 Security Features

- **JWT-based authentication** with refresh tokens
- **Role-based access control** (RBAC)
- **Rate limiting** and brute force protection
- **Input validation** and sanitization
- **CORS configuration** per environment
- **Security headers** (HSTS, CSP, etc.)
- **Sensitive data filtering** in logs
- **SQL injection prevention**
- **XSS protection**

## 📈 Scalability

- **Horizontal scaling** via Docker replicas
- **Database connection pooling**
- **Redis caching** for session management
- **Async message processing** with RabbitMQ
- **Load balancing** through Kong Gateway
- **Stateless service design**

## 🐛 Troubleshooting

### Common Issues

1. **Service won't start**: Check Docker resources and port conflicts
2. **Database connection failed**: Verify PostgreSQL is running
3. **JWT token invalid**: Check SECRET_KEY consistency
4. **Rate limit exceeded**: Adjust rate limiting configuration
5. **AI service timeout**: Increase model loading timeout

### Debug Commands

```bash
# Check service status
docker-compose -f docker-compose.microservices.yml ps

# View service logs
make logs

# Test database connection
make test-local-db

# Check Kong configuration
curl http://localhost:8101/services
```

This comprehensive overview provides everything needed to understand, deploy, and work with the Eindr microservices platform.