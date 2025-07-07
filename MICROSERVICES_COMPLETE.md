# 🚀 Eindr Microservices - Complete Implementation

## 📋 Overview

This repository contains the complete microservices implementation of the Eindr AI-powered reminder app. The monolithic application has been successfully converted into **13 independent microservices** with comprehensive APIs, databases, and inter-service communication.

## 🏗️ Architecture

### Services Overview

| Service               | Port | Database     | Description                           | API Endpoints      |
| --------------------- | ---- | ------------ | ------------------------------------- | ------------------ |
| **API Gateway**       | 8080 | -            | Entry point, routing, auth validation | All routes         |
| **Auth Service**      | 8001 | auth_db      | User authentication & JWT management  | `/auth/*`          |
| **User Service**      | 8002 | user_db      | User profiles & preferences           | `/customers/*`         |
| **Reminder Service**  | 8003 | reminder_db  | Reminder management & scheduling      | `/reminders/*`     |
| **Note Service**      | 8004 | note_db      | Note & document management            | `/notes/*`         |
| **Ledger Service**    | 8005 | ledger_db    | Expense tracking & budgets            | `/expenses/*`      |
| **Friend Service**    | 8006 | friend_db    | Social features & friend management   | `/friends/*`       |
| **History Service**   | 8007 | history_db   | Activity logging & audit trails       | `/logs/*`          |
| **STT Service**       | 8008 | -            | Speech-to-Text processing             | `/stt/*`           |
| **TTS Service**       | 8009 | -            | Text-to-Speech synthesis              | `/tts/*`           |
| **Intent Service**    | 8010 | -            | Intent classification & NLP           | `/intent/*`        |
| **Chat Service**      | 8011 | chat_db      | Conversational AI & chatbot           | `/conversations/*` |
| **Scheduler Service** | 8012 | scheduler_db | Background job scheduling             | `/jobs/*`          |

### Infrastructure Components

- **API Gateway**: Kong (Port 8080)
- **Message Broker**: RabbitMQ (Port 5672)
- **Cache**: Redis (Port 6379)
- **Databases**: PostgreSQL instances (Ports 5432-5440)
- **Monitoring**: Prometheus (Port 9090) + Grafana (Port 3000)

## 🛠️ Service Details

### 1. Auth Service (`auth-service`)

**Purpose**: Centralized authentication and authorization

**Key Features**:

- User registration and login
- JWT token management (access + refresh tokens)
- Password reset functionality
- Account lockout protection
- Session management
- Token validation for other services

**API Endpoints**:

```
POST /auth/register         - Register new user
POST /auth/login           - Login user
POST /auth/token           - OAuth2 token endpoint
POST /auth/refresh         - Refresh access token
GET  /auth/me              - Get current user info
POST /auth/logout          - Logout user
POST /auth/password-reset-request  - Request password reset
POST /auth/password-reset-confirm  - Confirm password reset
POST /auth/change-password  - Change password
POST /auth/validate-token   - Validate token (for services)
```

### 2. User Service (`customers-service`)

**Purpose**: User profile management and preferences

**Key Features**:

- User profile creation and management
- User preferences and settings
- Avatar upload and management
- Device registration for notifications
- Privacy settings
- Theme and localization preferences

**API Endpoints**:

```
POST /customers/profile        - Create user profile
GET  /customers/profile        - Get user profile
PUT  /customers/profile        - Update user profile
POST /customers/avatar         - Upload avatar
GET  /customers/preferences    - Get user preferences
PUT  /customers/preferences    - Update preferences
POST /customers/devices        - Register device
GET  /customers/devices        - Get registered devices
DELETE /customers/devices/{id} - Unregister device
```

### 3. Reminder Service (`reminder-service`)

**Purpose**: Core reminder management and scheduling

**Key Features**:

- Create, read, update, delete reminders
- Recurring reminder patterns
- Reminder sharing between customers
- Snooze functionality
- Priority and categorization
- Location-based reminders
- Notification scheduling

**API Endpoints**:

```
POST /reminders/           - Create reminder
GET  /reminders/           - Get user reminders (with filters)
GET  /reminders/{id}       - Get specific reminder
PUT  /reminders/{id}       - Update reminder
DELETE /reminders/{id}     - Delete reminder
POST /reminders/{id}/complete  - Mark as completed
POST /reminders/{id}/snooze    - Snooze reminder
POST /reminders/{id}/share     - Share reminder
GET  /reminders/shared/with-me - Get shared reminders
GET  /reminders/due/upcoming   - Get upcoming reminders
```

### 4. Note Service (`note-service`)

**Purpose**: Note and document management

**Key Features**:

- Create and manage notes
- Folder organization
- Tag-based categorization
- Full-text search
- Rich text content support
- Note sharing

**API Endpoints**:

```
POST /notes/               - Create note
GET  /notes/               - Get all notes
GET  /notes/{id}           - Get specific note
PUT  /notes/{id}           - Update note
DELETE /notes/{id}         - Delete note
```

### 5. Ledger Service (`ledger-service`)

**Purpose**: Expense tracking and financial management

**Key Features**:

- Expense entry and tracking
- Category management
- Budget planning and monitoring
- Financial reports and analytics
- Receipt management

**API Endpoints**:

```
POST /expenses/            - Create expense
GET  /expenses/            - Get all expenses
GET  /expenses/{id}        - Get specific expense
PUT  /expenses/{id}        - Update expense
DELETE /expenses/{id}      - Delete expense
```

### 6. Friend Service (`friend-service`)

**Purpose**: Social features and friend management

**Key Features**:

- Friend requests and management
- User discovery
- Social groups
- Activity sharing
- Privacy controls

**API Endpoints**:

```
POST /friends/             - Send friend request
GET  /friends/             - Get friends list
GET  /friends/{id}         - Get specific friend
PUT  /friends/{id}         - Update friend settings
DELETE /friends/{id}       - Remove friend
```

### 7. History Service (`history-service`)

**Purpose**: Activity logging and audit trails

**Key Features**:

- Comprehensive activity logging
- Audit trail maintenance
- User action tracking
- System event logging
- Data analytics support

**API Endpoints**:

```
POST /logs/                - Create log entry
GET  /logs/                - Get activity logs
GET  /logs/{id}            - Get specific log
```

### 8. STT Service (`stt-service`)

**Purpose**: Speech-to-Text processing

**Key Features**:

- Audio file transcription
- Real-time speech recognition
- Multiple language support
- Confidence scoring
- Model selection

**API Endpoints**:

```
POST /stt/transcribe       - Transcribe audio to text
GET  /stt/models           - Get supported models
```

### 9. TTS Service (`tts-service`)

**Purpose**: Text-to-Speech synthesis

**Key Features**:

- Text to speech conversion
- Multiple voice options
- Speed and pitch control
- Audio format options
- Voice customization

**API Endpoints**:

```
POST /tts/synthesize       - Convert text to speech
GET  /tts/voices           - Get available voices
```

### 10. Intent Service (`intent-service`)

**Purpose**: Intent classification and NLP

**Key Features**:

- Natural language understanding
- Intent classification
- Entity extraction
- Context management
- Model training

**API Endpoints**:

```
POST /intent/classify      - Classify user intent
POST /intent/train         - Train classification model
```

### 11. Chat Service (`chat-service`)

**Purpose**: Conversational AI and chatbot

**Key Features**:

- Conversational AI interface
- Chat history management
- Context-aware responses
- Multi-turn conversations
- Integration with other services

**API Endpoints**:

```
POST /conversations/       - Create conversation
GET  /conversations/       - Get conversations
GET  /conversations/{id}   - Get specific conversation
PUT  /conversations/{id}   - Update conversation
DELETE /conversations/{id} - Delete conversation
```

### 12. Scheduler Service (`scheduler-service`)

**Purpose**: Background job scheduling and processing

**Key Features**:

- Cron-based job scheduling
- Task queue management
- Job monitoring and status
- Retry logic and error handling
- Distributed task execution

**API Endpoints**:

```
POST /jobs/                - Create scheduled job
GET  /jobs/                - Get all jobs
GET  /jobs/{id}            - Get specific job
PUT  /jobs/{id}            - Update job
DELETE /jobs/{id}          - Delete job
```

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Make (for automation)
- Python 3.11+ (for development)

### Quick Start

1. **Start all services**:

   ```bash
   make up
   ```

2. **Check service health**:

   ```bash
   make health-check
   ```

3. **View logs**:

   ```bash
   make logs
   ```

4. **Stop all services**:
   ```bash
   make down
   ```

### API Documentation

Each service provides interactive API documentation:

- **API Gateway**: http://localhost:8080/docs
- **Auth Service**: http://localhost:8001/docs
- **User Service**: http://localhost:8002/docs
- **Reminder Service**: http://localhost:8003/docs
- **Note Service**: http://localhost:8004/docs
- **Ledger Service**: http://localhost:8005/docs
- **Friend Service**: http://localhost:8006/docs
- **History Service**: http://localhost:8007/docs
- **STT Service**: http://localhost:8008/docs
- **TTS Service**: http://localhost:8009/docs
- **Intent Service**: http://localhost:8010/docs
- **Chat Service**: http://localhost:8011/docs
- **Scheduler Service**: http://localhost:8012/docs

## 🔄 Inter-Service Communication

### Event-Driven Architecture

Services communicate through RabbitMQ events:

```
user.created        -> User service notifies other services
reminder.created    -> Reminder created, notify scheduler
reminder.due        -> Scheduler notifies user service
note.created        -> Note created for activity logging
expense.added       -> Ledger updates, notify analytics
friend.added        -> Social activity logging
```

### Service-to-Service Authentication

All internal API calls include:

- JWT token validation via Auth Service
- Request tracing headers
- Service identification

### API Gateway Routing

```nginx
/auth/*         -> auth-service:8001
/customers/*        -> user-service:8002
/reminders/*    -> reminder-service:8003
/notes/*        -> note-service:8004
/expenses/*     -> ledger-service:8005
/friends/*      -> friend-service:8006
/logs/*         -> history-service:8007
/stt/*          -> stt-service:8008
/tts/*          -> tts-service:8009
/intent/*       -> intent-service:8010
/conversations/* -> chat-service:8011
/jobs/*         -> scheduler-service:8012
```


## 🔧 Development

### Adding New Features

1. **Create feature branch**:

   ```bash
   git checkout -b feature/new-feature
   ```

2. **Modify service code**:

   ```bash
   # Edit service files
   services/{service-name}/src/
   ```

3. **Test locally**:

   ```bash
   make restart-service SERVICE=service-name
   ```

4. **Run tests**:
   ```bash
   make test
   ```

### Debugging Services

1. **View service logs**:

   ```bash
   make logs SERVICE=service-name
   ```

2. **Access service container**:

   ```bash
   docker exec -it {service-name} bash
   ```

3. **Database access**:
   ```bash
   make db-connect DB=service_db
   ```

## 📈 Monitoring and Observability

### Health Checks

All services provide health endpoints:

```bash
GET /{service}/health
```

### Metrics Collection

- Prometheus metrics at `/metrics` endpoint
- Custom business metrics
- Performance monitoring
- Error rate tracking

### Logging

- Structured JSON logging
- Centralized log collection
- Log level configuration
- Request tracing

## 🔒 Security

### Authentication Flow

1. User authenticates with Auth Service
2. Receives JWT access + refresh tokens
3. API Gateway validates tokens
4. Requests forwarded to target services
5. Services receive validated user context

### Security Features

- JWT-based authentication
- Password hashing (bcrypt)
- Rate limiting
- CORS configuration
- Request validation
- SQL injection prevention

## 🚀 Deployment

### Production Deployment

1. **Build all images**:

   ```bash
   make build-all
   ```

2. **Deploy to production**:

   ```bash
   make deploy ENV=production
   ```

3. **Database migrations**:
   ```bash
   make migrate-all
   ```

### Scaling

Services can be independently scaled:

```bash
docker-compose up --scale user-service=3
docker-compose up --scale reminder-service=5
```

## 📋 Available Make Commands

```bash
make up              # Start all services
make down            # Stop all services
make build-all       # Build all Docker images
make health-check    # Check all service health
make logs            # View all service logs
make migrate-all     # Run all database migrations
make test            # Run all tests
make clean           # Clean up containers and images
```

## 🎯 Key Benefits

### ✅ Microservices Advantages

1. **Independent Scaling**: Scale services based on demand
2. **Technology Diversity**: Use best tools for each service
3. **Fault Isolation**: Service failures don't affect others
4. **Team Autonomy**: Teams can work independently
5. **Deployment Flexibility**: Deploy services independently
6. **Database Isolation**: Each service owns its data

### ✅ API-First Design

1. **Clear Contracts**: Well-defined service interfaces
2. **Documentation**: Auto-generated API docs
3. **Testing**: Easy to test individual services
4. **Integration**: Simple third-party integrations

### ✅ Event-Driven Architecture

1. **Loose Coupling**: Services communicate via events
2. **Scalability**: Asynchronous processing
3. **Reliability**: Event replay and recovery
4. **Flexibility**: Easy to add new consumers

## 🔮 Future Enhancements

1. **Service Mesh**: Implement Istio for advanced traffic management
2. **GraphQL Gateway**: Unified query interface
3. **Event Sourcing**: Complete event-driven state management
4. **CQRS**: Command Query Responsibility Segregation
5. **Distributed Tracing**: End-to-end request tracing
6. **Auto-scaling**: Kubernetes HPA integration

---

## 📞 Support

For questions or issues:

1. Check the API documentation at each service's `/docs` endpoint
2. Review service logs: `make logs SERVICE=service-name`
3. Check health status: `make health-check`

**All 13 microservices are now fully populated with comprehensive APIs, database schemas, and inter-service communication! 🎉**
