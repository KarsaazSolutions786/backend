#!/bin/bash

# ===============================================
# EINDR MICROSERVICES DATABASE SETUP SCRIPT
# ===============================================

echo "🚀 Setting up Eindr Microservices Databases..."
echo "=============================================="

# Set PostgreSQL binary paths
PG_BIN_PATH="/usr/local/Cellar/postgresql@15/15.13/bin"
PSQL="$PG_BIN_PATH/psql"
PG_ISREADY="$PG_BIN_PATH/pg_isready"

# Check if PostgreSQL is running
if ! $PG_ISREADY > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not running. Please start PostgreSQL first."
    echo "   On macOS: brew services start postgresql@15"
    echo "   On Ubuntu: sudo service postgresql start"
    exit 1
fi

echo "✅ PostgreSQL is running"

# Get database connection info
DB_HOST=${DB_HOST:-localhost}
DB_USER=${DB_USER:-$(whoami)}
DB_PASSWORD=${DB_PASSWORD:-""}

echo "📊 Database Configuration:"
echo "   Host: $DB_HOST"
echo "   User: $DB_USER"
echo ""

# Function to run SQL with error handling
run_sql() {
    local sql_file=$1
    echo "📝 Executing: $sql_file"
    
    if [ -n "$DB_PASSWORD" ]; then
        PGPASSWORD=$DB_PASSWORD $PSQL -h $DB_HOST -U $DB_USER -f $sql_file
    else
        $PSQL -h $DB_HOST -U $DB_USER -f $sql_file
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully executed: $sql_file"
    else
        echo "❌ Failed to execute: $sql_file"
        exit 1
    fi
    echo ""
}

# Create databases and tables
echo "🏗️  Creating databases and tables..."
run_sql setup_databases.sql

# Verify databases were created
echo "🔍 Verifying database creation..."
if [ -n "$DB_PASSWORD" ]; then
    PGPASSWORD=$DB_PASSWORD $PSQL -h $DB_HOST -U $DB_USER -c "
    SELECT 
        datname as database_name, 
        pg_size_pretty(pg_database_size(datname)) as size
    FROM pg_database 
    WHERE datname IN ('auth_db', 'user_db', 'reminder_db', 'note_db', 'ledger_db', 
                      'friend_db', 'history_db', 'chat_db', 'kong_db')
    ORDER BY datname;"
else
    $PSQL -h $DB_HOST -U $DB_USER -c "
    SELECT 
        datname as database_name, 
        pg_size_pretty(pg_database_size(datname)) as size
    FROM pg_database 
    WHERE datname IN ('auth_db', 'user_db', 'reminder_db', 'note_db', 'ledger_db', 
                      'friend_db', 'history_db', 'chat_db', 'kong_db')
    ORDER BY datname;"
fi

echo ""
echo "🎉 DATABASE SETUP COMPLETED!"
echo "=============================="
echo ""
echo "📋 Created Databases on Local PostgreSQL:"
echo "   1. auth_db - User authentication & sessions"
echo "   2. user_db - User profiles & preferences"  
echo "   3. reminder_db - Reminders & notifications"
echo "   4. note_db - Notes & folders"
echo "   5. ledger_db - Expenses & budgets"
echo "   6. friend_db - Friendships & permissions"
echo "   7. history_db - Activity logs & analytics"
echo "   8. chat_db - AI conversations"
echo "   9. kong_db - API Gateway configuration"
echo ""
echo "🔧 Next Steps:"
echo "   1. Test database connections: make test-databases"
echo "   2. For Docker deployment: make up"
echo "   3. For local development: Configure services to use localhost:5432"
echo ""
echo "🌐 Local Development URLs:"
echo "   - PostgreSQL: localhost:5432"
echo "   - Use DATABASE_URL: postgresql://$DB_USER@localhost:5432/[db_name]"

