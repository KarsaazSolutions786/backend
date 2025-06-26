#!/bin/bash

# ===============================================
# EINDR MICROSERVICES MIGRATION SCRIPT
# ===============================================

echo "🚀 Running Eindr Microservices Migrations..."
echo "============================================"

# Check if PostgreSQL is running
if ! pg_isready > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

echo "✅ PostgreSQL is running"

# Database connection settings
DB_HOST=${DB_HOST:-localhost}
DB_USER=${DB_USER:-$(whoami)}
DB_PASSWORD=${DB_PASSWORD:-""}

# Function to run migration with error handling
run_migration() {
    local migration_file=$1
    local service_name=$2
    
    echo "🔄 Running migration: $service_name"
    
    if [ -n "$DB_PASSWORD" ]; then
        PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -f $migration_file
    else
        psql -h $DB_HOST -U $DB_USER -f $migration_file
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ $service_name migration completed"
    else
        echo "❌ $service_name migration failed"
        exit 1
    fi
    echo ""
}

# Run all migrations in order
run_migration "database_migrations/001_auth_service.sql" "Auth Service"
run_migration "database_migrations/002_user_service.sql" "User Service"
run_migration "database_migrations/003_reminder_service.sql" "Reminder Service"
run_migration "database_migrations/004_note_service.sql" "Note Service"
run_migration "database_migrations/005_ledger_service.sql" "Ledger Service"
run_migration "database_migrations/006_friend_service.sql" "Friend Service"
run_migration "database_migrations/007_history_service.sql" "History Service"
run_migration "database_migrations/008_chat_service.sql" "Chat Service"

echo "🎉 ALL MIGRATIONS COMPLETED SUCCESSFULLY!"
echo "========================================"

# Verify table creation
echo "🔍 Verifying table creation..."

services=("auth_db" "user_db" "reminder_db" "note_db" "ledger_db" "friend_db" "history_db" "chat_db")

for db in "${services[@]}"; do
    echo "📊 Tables in $db:"
    if [ -n "$DB_PASSWORD" ]; then
        PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $db -c "
        SELECT schemaname, tablename, 
               pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY tablename;"
    else
        psql -h $DB_HOST -U $DB_USER -d $db -c "
        SELECT schemaname, tablename, 
               pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY tablename;"
    fi
    echo ""
done

echo "✅ Migration verification completed!"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start microservices: make up-microservices"
echo "   2. Test database connections in each service"
echo "   3. Use APIs through Kong Gateway: http://localhost:8080"

