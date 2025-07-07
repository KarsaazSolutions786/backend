# Eindr Database - Complete Guide

## 📈 Database Overview

The Eindr microservices architecture uses a **single PostgreSQL database** (`eindr_db`) that serves all 14 microservices. This unified approach simplifies development while maintaining service independence through proper data modeling and access patterns.

### 🏗️ Architecture Highlights

- **Database Engine**: PostgreSQL 15
- **Total Tables**: 27 tables
- **Services Supported**: 14 microservices
- **Data Model**: Relational with proper foreign key constraints
- **Hosting**: Docker containerized deployment

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- PostgreSQL client tools (optional)
- pgAdmin access (optional)

### Start Database

```bash
# Start the complete microservices stack (includes database)
make up

# Or start just the database
docker-compose -f docker-compose.microservices.yml up -d new-postgres-server

# Verify database is running
docker ps | grep postgres
```

### Connection Details

| Environment | Host | Port | Database | Username | Password |
|-------------|------|------|----------|----------|----------|
| **Docker Internal** | `new-postgres-server` | `5432` | `eindr_db` | `eindr` | `eindr_pass` |
| **External/Host** | `localhost` | `5433` | `eindr_db` | `eindr` | `eindr_pass` |

### Connection Strings

```bash
# For microservices (internal Docker network)
DATABASE_URL=postgresql://eindr:eindr_pass@new-postgres-server:5432/eindr_db

# For external tools/development
DATABASE_URL=postgresql://eindr:eindr_pass@localhost:5433/eindr_db
```

## 📝 Database Schema

### 📋 Complete Table List

#### 🔑 Authentication & User Management
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `customers` | User accounts and authentication | `id`, `email`, `password_hash`, `is_active` |
| `customer_sessions` | Active login sessions | `customer_id`, `session_token`, `expires_at` |
| `login_attempts` | Security audit log | `customer_id`, `email`, `is_success`, `ip_address` |
| `customers_profiles` | User profile information | `customer_id`, `full_name`, `bio`, `avatar_url` |
| `customer_preferences` | User settings and preferences | `customer_id`, `theme`, `notifications`, `language_id` |
| `customers_devices` | Registered devices for push notifications | `customer_id`, `device_token`, `device_type` |

#### ⏰ Reminders & Scheduling
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `reminders` | Core reminder data | `id`, `customer_id`, `title`, `time`, `repeat_pattern_id` |
| `reminder_shares` | Shared reminders between users | `reminder_id`, `owner_customer_id`, `shared_with_customer_id` |
| `reminder_notifications` | Notification delivery history | `reminder_id`, `customer_id`, `sent_at`, `status` |
| `priority_levels` | Reminder priority system | `id`, `level_name`, `sort_order` |

#### 📝 Notes & Documentation
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `notes` | User notes and documents | `id`, `customer_id`, `title`, `description`, `content_type` |
| `note_shares` | Shared notes between users | `note_id`, `owner_customer_id`, `shared_with_customer_id` |

#### 💰 Financial Tracking
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `ledger_entries` | Expense/income tracking | `id`, `customer_id`, `amount`, `ledger_direction_id` |
| `ledger_direction` | Income vs expense classification | `id`, `direction_name` |

#### 👥 Social Features
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `friendships` | User connections | `id`, `customer_id`, `friend_id`, `status` |
| `friend_permissions` | Sharing permissions between friends | `customer_id`, `friend_id`, `auto_accept_reminders` |
| `friend_request_history` | Friend request audit trail | `requester_id`, `requested_id`, `status`, `created_at` |

#### 🤖 AI & Chat
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `conversions` | AI chat conversations | `id`, `customer_id`, `title`, `is_active` |
| `chat_messages` | Individual chat messages | `id`, `conversions_id`, `role`, `description` |
| `api_usage_logs` | API usage tracking | `customer_id`, `end_point`, `status_code`, `response_time_ms` |

#### 🌍 Configuration & Localization
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `languages` | Supported languages | `id`, `code`, `name`, `native_name` |
| `timezones` | Timezone data | `id`, `name`, `gmt_offset` |
| `subscription_plans` | User subscription tiers | `id`, `name`, `price`, `features` |
| `condition_states` | System states | `id`, `label` |
| `label_codes` | Categorization codes | `id`, `code`, `description` |
| `label_groups` | Label grouping system | `id`, `group_name` |
| `language_label` | Localized labels | `language_id`, `label_code_id`, `value` |

## 🔄 Database Management

### Backup & Restore

```bash
# Create backup
./backup_eindr_db.sh

# Restore from backup
./restore_eindr_db.sh ./backups/eindr_db_backup_YYYYMMDD_HHMMSS.sql

# Verify database integrity
./verify_eindr_db.sh
```

### Health Checks

```bash
# Test database connection
docker exec backend-new-postgres-server-1 pg_isready -U eindr -d eindr_db

# Check database size
docker exec backend-new-postgres-server-1 psql -U eindr -d eindr_db -c "SELECT pg_size_pretty(pg_database_size('eindr_db'));"

# View active connections
docker exec backend-new-postgres-server-1 psql -U eindr -d eindr_db -c "SELECT count(*) FROM pg_stat_activity;"
```

### Useful Queries

```sql
-- View all tables and their row counts
SELECT schemaname,tablename,attname,n_distinct,correlation FROM pg_stats;

-- Check foreign key relationships
SELECT tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
WHERE constraint_type = 'FOREIGN KEY';

-- View database statistics
SELECT 
    table_name,
    pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY pg_total_relation_size(table_name::regclass) DESC;
```

## 🌐 pgAdmin Access

### Setup pgAdmin Connection

1. **Access pgAdmin**: http://localhost:5050
2. **Login**: 
   - Email: `admin@admin.com`
   - Password: `admin`
3. **Add Server**:
   - Name: `Eindr Database`
   - Host: `new-postgres-server`
   - Port: `5432`
   - Database: `eindr_db`
   - Username: `eindr`
   - Password: `eindr_pass`

### pgAdmin Features

- **Query Tool**: Write and execute custom SQL
- **Data Viewer**: Browse table data with pagination
- **ERD Tool**: Visualize table relationships
- **Import/Export**: Data management tools
- **Performance Dashboard**: Monitor database performance

## 🔗 Microservices Integration

### Service-Specific Database Access

Each microservice connects to the same database but focuses on specific tables:

| Service | Primary Tables | Access Pattern |
|---------|----------------|----------------|
| **auth-service** | `customers`, `customer_sessions`, `login_attempts` | Read/Write authentication data |
| **customer-service** | `customers_profiles`, `customer_preferences`, `customers_devices` | Manage user profiles |
| **reminder-service** | `reminders`, `reminder_shares`, `reminder_notifications` | Handle reminder operations |
| **note-service** | `notes`, `note_shares` | Manage notes and sharing |
| **ledger-service** | `ledger_entries`, `ledger_direction` | Track financial data |
| **friend-service** | `friendships`, `friend_permissions`, `friend_request_history` | Social features |
| **chat-service** | `conversions`, `chat_messages` | AI conversations |
| **history-service** | `api_usage_logs` | Activity logging |

### Database Connection Pool

Each service maintains its own connection pool:

```python
# Example connection configuration
DATABASE_URL = "postgresql://eindr:eindr_pass@new-postgres-server:5432/eindr_db"
SQLALCHEMY_DATABASE_URL = DATABASE_URL
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=300
)
```

## 📊 Performance & Optimization

### Indexes

The database includes optimized indexes for:

- **Primary Keys**: All tables have auto-incrementing integer PKs
- **Foreign Keys**: Proper relationships with indexed foreign keys
- **Email Lookups**: Unique index on `customers.email`
- **User Queries**: Composite indexes on frequently queried columns
- **Timestamps**: Indexes on `created_at` and `updated_at` columns

### Query Optimization Tips

```sql
-- Use EXPLAIN ANALYZE for query planning
EXPLAIN ANALYZE SELECT * FROM reminders WHERE customer_id = 1;

-- Monitor slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename = 'reminders';
```

## 🔒 Security & Access Control