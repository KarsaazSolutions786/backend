# Eindr - AI-Powered Personal Assistant Platform

## 🌟 Overview
Eindr is a sophisticated microservices-based platform that combines AI capabilities with personal productivity tools. It provides intelligent reminder management, social collaboration features, and comprehensive personal finance tracking.

## 📑 Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Services](#services)
- [Getting Started](#getting-started)
- [Development](#development)
- [Database](#database)
- [Monitoring](#monitoring)
- [Contributing](#contributing)

## ✨ Features

### 🤖 AI Capabilities
  - Speech-to-Text (Whisper STT)
  - Text-to-Speech (Coqui TTS)
- Intent Classification (MiniLM)
- Conversational AI Pipeline
- Multi-intent Processing

### 🎯 Core Features
- Smart Reminder Management
- Social Collaboration Tools
- Note Taking System
- Personal Finance Tracking
- Friend Network Management
- Multi-language Support
- Real-time Notifications

### 💼 Business Features
- SaaS Subscription Model
- Usage Analytics
- Multi-tenant Architecture
- Comprehensive Audit Logging

## 🏗️ Architecture

### Microservices
- **API Gateway**: Kong-based API management
- **Auth Service**: JWT-based authentication
- **Customer Service**: User management
- **AI Pipeline**: Orchestrates AI processing
- **Chat Service**: Manages conversations
- **Reminder Service**: Core reminder functionality
- **Friend Service**: Social features
- **Note Service**: Document management
- **Ledger Service**: Financial tracking
- **History Service**: Activity logging
- **STT/TTS Services**: Speech processing
- **Intent Service**: NLP processing
- **Scheduler Service**: Task scheduling

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- PostgreSQL 15
- Make

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/eindr.git
cd eindr

# Set up environment
cp local.env .env

# Start services
make up

# Verify deployment
make verify
```

### Environment Setup
```bash
# Start development environment
make dev

# Run database migrations
make migrate

# Seed test data
make seed
```

## 💻 Development

### Project Structure
```
backend/
├── services/           # Microservices
├── infrastructure/     # Infrastructure configs
├── scripts/           # Utility scripts
├── models/            # AI models
└── docker-compose.yml # Service orchestration
```

### Service Development
```bash
# Create new service
make create-service name=new-service

# Run tests
make test service=service-name

# Update service
make update service=service-name
```

### Git LFS Setup
This repository uses Git LFS to handle large ML model files. If you're setting up the project for the first time:

```bash
# Install Git LFS (if not already installed)
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install git-lfs
# Windows: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Pull LFS files
git lfs pull
```

For existing repositories with large model files in history, run the migration script:
```bash
./scripts/migrate_to_lfs.sh
```

## 🗄️ Database

### Overview
- Single PostgreSQL database
- 27 tables across services
- Proper foreign key constraints
- Containerized deployment

### Management
```bash
# Backup database
./backup_eindr_db.sh

# Restore database
./restore_eindr_db.sh ./backups/backup_name.sql

# Verify database
./verify_eindr_db.sh
```

## 📊 Monitoring

### Infrastructure
- Prometheus metrics
- Grafana dashboards
- Service health checks
- Performance monitoring

### Access Points
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- pgAdmin: http://localhost:5050

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Guidelines
- Follow PEP 8 style guide
- Write meaningful commit messages
- Add proper documentation
- Include unit tests

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation
- [API Guide](MICROSERVICES_API_GUIDE.md)
- [Database Guide](DATABASE_README.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [Microservices](MICROSERVICES_COMPLETE.md)