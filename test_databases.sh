#!/bin/bash

echo "🧪 Testing Database Connections..."
echo "================================="

# Set PostgreSQL binary paths
PG_BIN_PATH="/usr/local/Cellar/postgresql@15/15.13/bin"
PSQL="$PG_BIN_PATH/psql"

DB_HOST=${DB_HOST:-localhost}
DB_USER=${DB_USER:-$(whoami)}
DB_PASSWORD=${DB_PASSWORD:-""}

services=("auth_db" "user_db" "reminder_db" "note_db" "ledger_db" "friend_db" "history_db" "chat_db" "kong_db")

for db in "${services[@]}"; do
    echo -n "Testing $db... "
    
    if [ -n "$DB_PASSWORD" ]; then
        PGPASSWORD=$DB_PASSWORD $PSQL -h $DB_HOST -U $DB_USER -d $db -c "SELECT 1;" > /dev/null 2>&1
    else
        $PSQL -h $DB_HOST -U $DB_USER -d $db -c "SELECT 1;" > /dev/null 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Connected"
    else
        echo "❌ Failed"
    fi
done

echo ""
echo "🔍 Database Summary:"
if [ -n "$DB_PASSWORD" ]; then
    PGPASSWORD=$DB_PASSWORD $PSQL -h $DB_HOST -U $DB_USER -c "
    SELECT 
        datname as database, 
        pg_size_pretty(pg_database_size(datname)) as size,
        (SELECT count(*) FROM pg_stat_activity WHERE datname = pg_database.datname) as connections
    FROM pg_database 
    WHERE datname IN ('auth_db', 'user_db', 'reminder_db', 'note_db', 'ledger_db', 
                      'friend_db', 'history_db', 'chat_db', 'kong_db')
    ORDER BY datname;"
else
    $PSQL -h $DB_HOST -U $DB_USER -c "
    SELECT 
        datname as database, 
        pg_size_pretty(pg_database_size(datname)) as size,
        (SELECT count(*) FROM pg_stat_activity WHERE datname = pg_database.datname) as connections
    FROM pg_database 
    WHERE datname IN ('auth_db', 'user_db', 'reminder_db', 'note_db', 'ledger_db', 
                      'friend_db', 'history_db', 'chat_db', 'kong_db')
    ORDER BY datname;"
fi

