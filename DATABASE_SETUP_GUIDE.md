# Eindr Microservices - Local Database Setup Guide

## ✅ Database Setup Complete!

Your local PostgreSQL database is now properly configured and running in Docker. All 9 microservice databases have been created automatically.

## 📊 Database Configuration

- **Host**: localhost:5432
- **User**: eindr_user
- **Password**: eindr_pass
- **Container**: PostgreSQL 15 in Docker

## 🗄️ Created Databases

All databases are running in a single PostgreSQL instance:

1. **auth_db** - User authentication & sessions
2. **user_db** - User profiles & preferences
3. **reminder_db** - Reminders & notifications
4. **note_db** - Notes & folders
5. **ledger_db** - Expenses & budgets
6. **friend_db** - Friendships & permissions
7. **history_db** - Activity logs & analytics
8. **chat_db** - AI conversations
9. **kong_db** - API Gateway configuration

## 🚀 Quick Commands

### Start/Stop Database

```bash
# Start PostgreSQL
make start-local-db

# Stop PostgreSQL
make stop-local-db

# Reset database (destroys all data)
make reset-local-db
```

### Test & Monitor

```bash
# Test connection
make test-local-db

# View logs
make local-db-logs

# Check database status
python3 test_connection.py
```

### Database Management

```bash
# Start PgAdmin (optional GUI)
make start-pgadmin

# Access PgAdmin at: http://localhost:5050
# Email: admin@eindr.local
# Password: admin123
```

## 🔧 Connection Details for Services

Each microservice should use these database URLs:

```bash
# Auth Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/auth_db

# User Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/user_db

# Reminder Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/reminder_db

# Note Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/note_db

# Ledger Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/ledger_db

# Friend Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/friend_db

# History Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/history_db

# Chat Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/chat_db

# Scheduler Service
DATABASE_URL=postgresql://eindr_user:eindr_pass@localhost:5432/scheduler_db
```

## 📋 Environment Variables

Use the `local.env` file for local development:

```bash
# Copy environment variables
cp local.env .env

# Or source them directly
source local.env
```

## 🛠️ Troubleshooting

### Port 5432 Already in Use

```bash
# Stop any local PostgreSQL instances
brew services stop postgresql@15
sudo pkill -f postgres

# Then restart Docker PostgreSQL
make start-local-db
```

### Database Connection Issues

```bash
# Check if container is running
docker ps | grep postgres

# View logs for errors
make local-db-logs

# Test connection
python3 test_connection.py
```

### Reset Everything

```bash
# Complete reset (removes all data)
make reset-local-db
```

## 🎯 Next Steps

1. **For Local Development**: Configure your services to use the database URLs above
2. **For Docker Deployment**: Use `make up` to start all services with their own databases
3. **For Production**: Update connection strings for your production database

## 🔍 Database Schema

All tables and relationships are defined in `setup_databases.sql`. Each database contains:

- **Proper indexes** for performance
- **Foreign key constraints** for data integrity
- **JSON columns** for flexible data storage
- **UUID primary keys** for distributed systems
- **Timestamps** for audit trails

The schema supports all features like:

- User authentication & profiles
- Reminders with notifications & sharing
- Notes with folders & collaboration
- Expense tracking & budgets
- Friend management & permissions
- Activity logging & analytics
- AI conversations & chat history

---

**Your local PostgreSQL database is ready for development! 🚀**
