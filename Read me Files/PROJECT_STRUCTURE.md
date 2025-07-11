# Eindr - Clean Project Structure

## 🏗️ Architecture Overview
- **Type**: Microservices Architecture
- **API Gateway**: Kong Gateway 
- **Services**: 13 independent microservices
- **Databases**: 9 PostgreSQL databases
- **Message Queue**: RabbitMQ
- **Cache**: Redis

## 📁 Project Structure
```
eindr-backend/
├── services/                    # All microservices
│   ├── api-gateway/            # Kong configuration
│   ├── auth-service/           # Authentication & JWT
│   ├── user-service/           # User management
│   ├── reminder-service/       # Reminders & notifications
│   ├── note-service/           # Note management
│   ├── ledger-service/         # Expense tracking
│   ├── friend-service/         # Social features
│   ├── history-service/        # Activity logs
│   ├── stt-service/            # Speech-to-text
│   ├── tts-service/            # Text-to-speech
│   ├── intent-service/         # Intent classification
│   ├── chat-service/           # AI conversations
│   └── scheduler-service/      # Background jobs
├── docker-compose.microservices.yml  # Full stack deployment
├── Makefile                    # Development commands
├── README.md                   # Main documentation
└── .git/                       # Version control

```

## 🚀 Quick Start
```bash
# Start all services
make up-microservices

# Setup Kong API Gateway  
make kong-setup

# View service status
make status

# View logs
make logs
```

## 🌐 Access Points
- **API Gateway**: http://localhost:8080
- **Kong Admin**: http://localhost:8101-8103  
- **Individual Services**: Ports 8001-8012
- **Databases**: Ports 5432-5442

## 📚 Documentation
- `README.md` - Main project documentation
- `MICROSERVICES_COMPLETE.md` - Detailed service documentation
- `MICROSERVICES_ARCHITECTURE_COMPLETE.md` - Architecture deep-dive
