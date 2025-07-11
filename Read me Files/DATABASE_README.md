# Eindr Database - Complete Guide

## 📊 Database Overview

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

## 🗄️ Database Schema

### 📋 Complete Table List

#### 🔐 Authentication & User Management
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

#### 🌐 Configuration & Localization
| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `languages` | Supported languages | `id`, `code`, `name`, `native_name` |
| `timezones` | Timezone data | `id`, `name`, `gmt_offset` |
| `subscription_plans` | User subscription tiers | `id`, `name`, `price`, `features` |
| `condition_states` | System states | `id`, `label` |
| `label_codes` | Categorization codes | `id`, `code`, `description` |
| `label_groups` | Label grouping system | `id`, `group_name` |
| `language_label` | Localized labels | `language_id`, `label_code_id`, `value` |

## 🔧 Database Management

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

## 🔄 Microservices Integration

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

## 📈 Performance & Optimization

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

### Database Security

- **Dedicated User**: Service uses `eindr` user (not superuser)
- **Network Isolation**: Database only accessible within Docker network
- **Connection Encryption**: SSL/TLS for production deployments
- **Audit Logging**: All login attempts and API usage logged

### Access Patterns

```python
# Row-Level Security Example (when implemented)
-- Customers can only access their own data
CREATE POLICY customer_access ON reminders 
FOR ALL TO app_user 
USING (customer_id = current_user_id());

-- Friend access for shared content
CREATE POLICY friend_shared_access ON reminder_shares 
FOR SELECT TO app_user 
USING (shared_with_customer_id = current_user_id());
```

## 🔍 Monitoring & Alerting

### Database Metrics

Monitor these key metrics:

- **Connection Count**: `SELECT count(*) FROM pg_stat_activity;`
- **Database Size**: `SELECT pg_size_pretty(pg_database_size('eindr_db'));`
- **Table Sizes**: `SELECT pg_size_pretty(pg_total_relation_size('table_name'));`
- **Query Performance**: Use `pg_stat_statements` extension
- **Lock Monitoring**: Check `pg_locks` for blocking queries

### Prometheus Integration

The database is monitored via Prometheus:

```yaml
# prometheus.yml excerpt
- job_name: "postgres-exporter"
  static_configs:
    - targets: ["postgres-exporter:9187"]
```

## 🚨 Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check if container is running
docker ps | grep postgres

# Check logs
docker logs backend-new-postgres-server-1

# Restart database
docker-compose restart new-postgres-server
```

#### High Memory Usage
```sql
-- Check memory usage
SELECT name, setting FROM pg_settings WHERE name LIKE '%memory%';

-- Monitor active queries
SELECT pid, usename, application_name, state, query 
FROM pg_stat_activity 
WHERE state = 'active';
```

#### Slow Queries
```sql
-- Enable statement logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1 second

-- Reload configuration
SELECT pg_reload_conf();
```

### Recovery Procedures

#### Database Corruption
```bash
# Check database integrity
docker exec backend-new-postgres-server-1 pg_dump -U eindr -d eindr_db --schema-only > schema_check.sql

# Full backup and restore if needed
./backup_eindr_db.sh
./restore_eindr_db.sh latest_backup.sql
```

#### Data Recovery
```sql
-- Recover deleted data (if within transaction)
BEGIN;
-- Perform recovery operations
ROLLBACK; -- or COMMIT;

-- Point-in-time recovery (requires WAL archiving)
-- This would be configured in production
```

## 📚 Additional Resources

### Documentation Links

- **PostgreSQL 15 Docs**: https://www.postgresql.org/docs/15/
- **pgAdmin Documentation**: https://www.pgadmin.org/docs/
- **Docker PostgreSQL**: https://hub.docker.com/_/postgres
- **SQLAlchemy ORM**: https://docs.sqlalchemy.org/

### Development Tools

- **DBeaver**: Universal database tool
- **DataGrip**: JetBrains database IDE
- **TablePlus**: Modern database management tool
- **psql**: Command-line PostgreSQL client

### Database Migration Tools

- **Alembic**: SQLAlchemy migration tool (used by FastAPI services)
- **Flyway**: Database migration tool
- **Liquibase**: Database change management

## 📞 Support & Maintenance

### Regular Maintenance Tasks

```bash
# Weekly tasks
# 1. Create database backup
./backup_eindr_db.sh

# 2. Check database size and growth
docker exec backend-new-postgres-server-1 psql -U eindr -d eindr_db -c "SELECT pg_size_pretty(pg_database_size('eindr_db'));"

# 3. Monitor slow queries
# Use pgAdmin or custom monitoring scripts

# 4. Update statistics
docker exec backend-new-postgres-server-1 psql -U eindr -d eindr_db -c "ANALYZE;"
```

### Contact Information

For database-related issues:

1. **Check Logs**: `docker logs backend-new-postgres-server-1`
2. **Verify Connectivity**: Use health check scripts
3. **Review Metrics**: Check Grafana dashboard
4. **Backup Data**: Always backup before major changes

---

**Database Version**: PostgreSQL 15  
**Last Updated**: January 2025  
**Maintained By**: Eindr Development Team

> 💡 **Pro Tip**: Always test database changes in a development environment before applying to production! 