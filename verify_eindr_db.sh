 m#!/bin/bash

# ===============================================
# EINDR MICROSERVICES DATABASE VERIFICATION SCRIPT
# For new-postgres-server with eindr_db database
# ===============================================

echo "🚀 Verifying Eindr Microservices Tables in new-postgres-server..."
echo "=================================================================="

# Database connection settings - connect via localhost with port mapping
DB_HOST="localhost"
DB_PORT="5500"  # Port mapping from new-postgres-server container
DB_NAME="eindr_db"
DB_USER="eindr"
DB_PASSWORD="eindr_pass"

echo "📋 Connection Details:"
echo "   Host: $DB_HOST"
echo "   Port: $DB_PORT"
echo "   Database: $DB_NAME"
echo "   User: $DB_USER"
echo ""

# Check if we can connect to the database
echo "🔍 Testing database connection..."
export PGPASSWORD="$DB_PASSWORD"

if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" > /dev/null 2>&1; then
    echo "❌ Cannot connect to PostgreSQL server at $DB_HOST:$DB_PORT"
    echo "   Please make sure the new-postgres-server container is running"
    exit 1
fi

echo "✅ Database connection successful"

# Function to check if table exists and show structure
check_table() {
    local table_name="$1"
    local service_name="$2"
    
    echo "🔍 Checking table: $table_name ($service_name)"
    
    # Check if table exists
    local table_exists=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = '$table_name'
    );")
    
    if [[ "$table_exists" == *"t"* ]]; then
        echo "✅ Table '$table_name' exists"
        
        # Show column count
        local col_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = '$table_name';")
        
        echo "   📊 Columns: $col_count"
    else
        echo "❌ Table '$table_name' does not exist"
        return 1
    fi
}

echo "📦 Verifying tables in eindr_db database..."
echo ""

# Core tables verification
echo "🔐 Auth Service Tables:"
check_table "customers" "Auth Service - Customers"
check_table "customer_sessions" "Auth Service - Sessions"
check_table "login_attempts" "Auth Service - Login Attempts"
check_table "subscription_plans" "Auth Service - Subscription Plans"
echo ""

echo "👤 User Service Tables:"
check_table "customers_profiles" "User Service - Customer Profiles"
check_table "customer_preferences" "User Service - Customer Preferences"
check_table "customers_devices" "User Service - Customer Devices"
check_table "timezones" "User Service - Timezones"
check_table "languages" "User Service - Languages"
echo ""

echo "📝 Note Service Tables:"
check_table "notes" "Note Service - Notes"
check_table "note_shares" "Note Service - Note Shares"
echo ""

echo "⏰ Reminder Service Tables:"
check_table "reminders" "Reminder Service - Reminders"
check_table "reminder_shares" "Reminder Service - Reminder Shares"
check_table "reminder_notifications" "Reminder Service - Notifications"
check_table "priority_levels" "Reminder Service - Priority Levels"
echo ""

echo "👥 Friend Service Tables:"
check_table "friendships" "Friend Service - Friendships"
check_table "friend_permissions" "Friend Service - Friend Permissions"
check_table "friend_request_history" "Friend Service - Friend Request History"
echo ""

echo "💰 Ledger Service Tables:"
check_table "ledger_entries" "Ledger Service - Ledger Entries"
check_table "ledger_direction" "Ledger Service - Ledger Direction"
echo ""

echo "💬 Chat Service Tables:"
check_table "conversions" "Chat Service - Conversations"
check_table "chat_messages" "Chat Service - Chat Messages"
echo ""

echo "📊 Support Tables:"
check_table "api_usage_logs" "History Service - API Usage Logs"
check_table "condition_states" "Support - Condition States"
check_table "label_codes" "Support - Label Codes"
check_table "language_label" "Support - Language Labels"
check_table "label_groups" "Support - Label Groups"
echo ""

echo "🎉 TABLE VERIFICATION COMPLETED!"
echo "================================="

# Get comprehensive table information
echo "🔍 Database Summary:"
echo ""

# Count all tables
tables_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) 
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
")

echo "📊 Total tables in database: $tables_count"
echo ""

# List all tables
echo "📋 Complete table list:"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT schemaname, tablename 
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY tablename;
"

echo ""
echo "🔗 Foreign Key Relationships:"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT 
    tc.table_name, 
    kcu.column_name, 
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name 
FROM 
    information_schema.table_constraints AS tc 
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' 
AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;
"

echo ""
echo "✅ Verification completed!"
echo ""
echo "🚀 Next Steps:"
echo "   1. Update microservice configurations to use:"
echo "      - Host: new-postgres-server (from containers) or localhost:5500 (from host)"
echo "      - Database: eindr_db"
echo "      - User: eindr"
echo "      - Password: eindr_pass"
echo "   2. Rebuild and restart microservices"
echo "   3. Test database connections in each service"
echo "   4. Use APIs through Kong Gateway: http://localhost:8080"
echo ""
echo "📝 Connection strings for services:"
echo "   Docker containers: postgresql://eindr:eindr_pass@new-postgres-server:5432/eindr_db"
echo "   From host machine: postgresql://eindr:eindr_pass@localhost:5500/eindr_db" 