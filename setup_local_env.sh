#!/bin/bash

# ===============================================
# EINDR MICROSERVICES - LOCAL ENVIRONMENT SETUP
# ===============================================
# Run this script to set up environment variables for local development
# Usage: source setup_local_env.sh

echo "🔧 Setting up local environment variables for Eindr microservices..."

# Database Configuration
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=eindr_user
export DB_PASSWORD=eindr_pass

# Individual Database URLs for each service
export AUTH_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/auth_db
export USER_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/user_db
export REMINDER_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/reminder_db
export NOTE_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/note_db
export LEDGER_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/ledger_db
export FRIEND_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/friend_db
export HISTORY_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/history_db
export CHAT_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/chat_db
export SCHEDULER_DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/scheduler_db

# Generic DATABASE_URL (for services that use this env var name)
export DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/eindr_dev

# Redis Configuration
export REDIS_URL=redis://localhost:6379

# RabbitMQ Configuration  
export RABBITMQ_URL=amqp://eindr:eindr123@localhost:5672/

# Service URLs (for local development)
export AUTH_SERVICE_URL=http://localhost:8001
export USER_SERVICE_URL=http://localhost:8002
export REMINDER_SERVICE_URL=http://localhost:8003
export NOTE_SERVICE_URL=http://localhost:8004
export LEDGER_SERVICE_URL=http://localhost:8005
export FRIEND_SERVICE_URL=http://localhost:8006
export HISTORY_SERVICE_URL=http://localhost:8007
export STT_SERVICE_URL=http://localhost:8008
export TTS_SERVICE_URL=http://localhost:8009
export INTENT_SERVICE_URL=http://localhost:8010
export CHAT_SERVICE_URL=http://localhost:8011
export SCHEDULER_SERVICE_URL=http://localhost:8012

# JWT Configuration
export JWT_SECRET=your-local-jwt-secret-key-for-development

# Development Settings
export DEBUG=true
export ENVIRONMENT=local

echo "✅ Environment variables set!"
echo ""
echo "📊 Database connections ready:"
echo "   - PostgreSQL: localhost:5432"
echo "   - User: eindr_user / Password: eindr_pass"
echo ""
echo "🗄️  Available databases:"
echo "   - auth_db, user_db, reminder_db, note_db"
echo "   - ledger_db, friend_db, history_db, chat_db"
echo "   - scheduler_db, kong_db"
echo ""
echo "🚀 Ready for local development!" 