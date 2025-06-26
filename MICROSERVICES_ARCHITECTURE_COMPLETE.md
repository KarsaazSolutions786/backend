# Eindr Backend - Complete Microservices Architecture

## Architecture Overview

Converting the existing monolith into **13 microservices** with dedicated databases and clear service boundaries.

### Service Inventory

| #   | Service               | Database       | Port | Responsibility                          |
| --- | --------------------- | -------------- | ---- | --------------------------------------- |
| 1   | **API Gateway**       | None           | 8080 | Routing, Rate limiting, Auth validation |
| 2   | **Auth Service**      | `auth_db`      | 8001 | User authentication, JWT tokens         |
| 3   | **User Service**      | `user_db`      | 8002 | User profiles, preferences              |
| 4   | **Reminder Service**  | `reminder_db`  | 8003 | Reminder CRUD, scheduling               |
| 5   | **Note Service**      | `note_db`      | 8004 | Note management                         |
| 6   | **Ledger Service**    | `ledger_db`    | 8005 | Expense tracking, IOUs                  |
| 7   | **Friend Service**    | `friend_db`    | 8006 | Friend management, permissions          |
| 8   | **History Service**   | `history_db`   | 8007 | Audit logging                           |
| 9   | **STT Service**       | None           | 8008 | Speech-to-Text processing               |
| 10  | **TTS Service**       | None           | 8009 | Text-to-Speech generation               |
| 11  | **Intent Service**    | None           | 8010 | Intent classification                   |
| 12  | **Chat Service**      | `chat_db`      | 8011 | Conversational AI                       |
| 13  | **Scheduler Service** | `scheduler_db` | 8012 | Background job processing               |

---

## Database Schema Design

### 1. Auth Service Database (`auth_db`)

```sql
-- Users table (authentication only)
CREATE TABLE users (
    id VARCHAR PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    password_reset_token VARCHAR,
    password_reset_expires TIMESTAMP,
    email_verification_token VARCHAR,
    email_verification_expires TIMESTAMP,
    last_login_ip VARCHAR,
    last_user_agent TEXT
);

-- Refresh tokens for JWT authentication
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL REFERENCES users(id),
    token_hash VARCHAR UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    revoked_at TIMESTAMP,
    device_info VARCHAR
);

-- Login attempt tracking
CREATE TABLE login_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR NOT NULL,
    ip_address VARCHAR NOT NULL,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    failure_reason VARCHAR,
    attempted_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_users_email_active ON users(email, is_active);
CREATE INDEX idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX idx_login_attempts_email_time ON login_attempts(email, attempted_at);
```

### 2. User Service Database (`user_db`)

```sql
-- User profiles (non-auth data)
CREATE TABLE user_profiles (
    user_id VARCHAR PRIMARY KEY, -- References auth_service.users.id
    display_name VARCHAR,
    first_name VARCHAR,
    last_name VARCHAR,
    avatar_url VARCHAR,
    language VARCHAR DEFAULT 'en',
    timezone VARCHAR DEFAULT 'UTC',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User preferences
CREATE TABLE user_preferences (
    user_id VARCHAR PRIMARY KEY REFERENCES user_profiles(user_id),
    allow_friends BOOLEAN DEFAULT TRUE,
    receive_shared_notes BOOLEAN DEFAULT TRUE,
    notification_sound VARCHAR DEFAULT 'default',
    tts_language VARCHAR DEFAULT 'en',
    chat_history_enabled BOOLEAN DEFAULT TRUE,
    theme VARCHAR DEFAULT 'light',
    email_notifications BOOLEAN DEFAULT TRUE,
    push_notifications BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User devices for push notifications
CREATE TABLE user_devices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL REFERENCES user_profiles(user_id),
    device_token VARCHAR NOT NULL,
    device_type VARCHAR NOT NULL, -- 'ios', 'android', 'web'
    device_name VARCHAR,
    is_active BOOLEAN DEFAULT TRUE,
    registered_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_devices_user_id ON user_devices(user_id);
```

### 3. Reminder Service Database (`reminder_db`)

```sql
-- Main reminders table
CREATE TABLE reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL, -- References auth_service.users.id
    title TEXT,
    description TEXT,
    time TIMESTAMP,
    repeat_pattern VARCHAR DEFAULT 'none', -- 'none', 'daily', 'weekly', 'monthly'
    timezone VARCHAR DEFAULT 'UTC',
    is_shared BOOLEAN DEFAULT FALSE,
    created_by VARCHAR, -- References auth_service.users.id
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    next_occurrence TIMESTAMP,
    occurrence_count INTEGER DEFAULT 0,
    max_occurrences INTEGER,
    priority VARCHAR DEFAULT 'medium', -- 'low', 'medium', 'high', 'urgent'
    category VARCHAR,
    tags TEXT[] -- Array of tags
);

-- Reminder notifications tracking
CREATE TABLE reminder_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id UUID NOT NULL REFERENCES reminders(id),
    user_id VARCHAR NOT NULL,
    notification_type VARCHAR NOT NULL, -- 'push', 'email', 'sms'
    status VARCHAR NOT NULL, -- 'pending', 'sent', 'delivered', 'failed'
    scheduled_at TIMESTAMP NOT NULL,
    sent_at TIMESTAMP,
    delivered_at TIMESTAMP,
    delivery_attempts INTEGER DEFAULT 0,
    last_attempt_at TIMESTAMP,
    failure_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Reminder sharing
CREATE TABLE reminder_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id UUID NOT NULL REFERENCES reminders(id),
    owner_user_id VARCHAR NOT NULL,
    shared_with_user_id VARCHAR NOT NULL,
    can_edit BOOLEAN DEFAULT FALSE,
    can_complete BOOLEAN DEFAULT TRUE,
    can_reschedule BOOLEAN DEFAULT FALSE,
    status VARCHAR DEFAULT 'pending', -- 'pending', 'accepted', 'declined'
    shared_at TIMESTAMP DEFAULT NOW(),
    responded_at TIMESTAMP
);

-- Embeddings for AI search
CREATE TABLE reminder_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id UUID NOT NULL REFERENCES reminders(id),
    user_id VARCHAR NOT NULL,
    embedding FLOAT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_reminders_user_time ON reminders(user_id, time);
CREATE INDEX idx_reminders_next_occurrence ON reminders(next_occurrence);
CREATE INDEX idx_reminder_notifications_pending ON reminder_notifications(status, scheduled_at);
```

### 4. Note Service Database (`note_db`)

```sql
-- Notes table
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL, -- References auth_service.users.id
    title VARCHAR,
    content TEXT NOT NULL,
    source VARCHAR DEFAULT 'manual', -- 'voice_input', 'manual', 'ai_generated'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_archived BOOLEAN DEFAULT FALSE,
    is_pinned BOOLEAN DEFAULT FALSE,
    category VARCHAR,
    tags TEXT[],
    color VARCHAR DEFAULT '#ffffff'
);

-- Note sharing
CREATE TABLE note_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    note_id UUID NOT NULL REFERENCES notes(id),
    owner_user_id VARCHAR NOT NULL,
    shared_with_user_id VARCHAR NOT NULL,
    can_edit BOOLEAN DEFAULT FALSE,
    can_view BOOLEAN DEFAULT TRUE,
    shared_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR DEFAULT 'pending'
);

-- Note attachments
CREATE TABLE note_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    note_id UUID NOT NULL REFERENCES notes(id),
    filename VARCHAR NOT NULL,
    file_path VARCHAR NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_notes_user_id ON notes(user_id);
CREATE INDEX idx_notes_created_at ON notes(created_at DESC);
CREATE INDEX idx_notes_category ON notes(category, user_id);
```

### 5. Ledger Service Database (`ledger_db`)

```sql
-- Ledger entries
CREATE TABLE ledger_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL, -- References auth_service.users.id
    contact_name VARCHAR,
    contact_user_id VARCHAR, -- References auth_service.users.id if contact is also a user
    amount DECIMAL(10,2) NOT NULL,
    direction VARCHAR CHECK (direction IN ('owe', 'owed')),
    description TEXT,
    category VARCHAR,
    currency VARCHAR DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_settled BOOLEAN DEFAULT FALSE,
    settled_at TIMESTAMP,
    settlement_note TEXT
);

-- Expense tracking (new feature)
CREATE TABLE expenses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    description TEXT,
    category VARCHAR,
    currency VARCHAR DEFAULT 'USD',
    receipt_url VARCHAR,
    created_by VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    expense_date DATE DEFAULT CURRENT_DATE
);

-- Contacts for ledger
CREATE TABLE contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    email VARCHAR,
    phone VARCHAR,
    user_id_ref VARCHAR, -- If contact is also a user
    created_by VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_ledger_entries_user_id ON ledger_entries(user_id);
CREATE INDEX idx_ledger_entries_contact ON ledger_entries(contact_name, user_id);
CREATE INDEX idx_expenses_user_date ON expenses(user_id, expense_date);
```

### 6. Friend Service Database (`friend_db`)

```sql
-- Friendships
CREATE TABLE friendships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL, -- References auth_service.users.id
    friend_id VARCHAR NOT NULL, -- References auth_service.users.id
    status VARCHAR CHECK (status IN ('pending', 'accepted', 'blocked')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, friend_id)
);

-- Friend permissions
CREATE TABLE friend_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    friend_id VARCHAR NOT NULL,
    auto_accept_reminders BOOLEAN DEFAULT FALSE,
    auto_accept_notes BOOLEAN DEFAULT FALSE,
    can_view_schedule BOOLEAN DEFAULT FALSE,
    can_create_reminders BOOLEAN DEFAULT FALSE,
    can_view_ledger BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, friend_id)
);

-- Friend requests history
CREATE TABLE friend_request_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requester_id VARCHAR NOT NULL,
    requested_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL, -- 'sent', 'accepted', 'declined', 'blocked'
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_friendships_user_id ON friendships(user_id);
CREATE INDEX idx_friendships_status ON friendships(status);
CREATE INDEX idx_friend_permissions_user_friend ON friend_permissions(user_id, friend_id);
```

### 7. History Service Database (`history_db`)

```sql
-- Activity logs (append-only)
CREATE TABLE history_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL, -- References auth_service.users.id
    action_type VARCHAR NOT NULL, -- 'reminder_created', 'note_added', 'friend_added', etc.
    entity_type VARCHAR, -- 'reminder', 'note', 'friend', etc.
    entity_id VARCHAR, -- ID of the affected entity
    content TEXT,
    interaction_type VARCHAR, -- 'chit_chat', 'stt', 'command', 'ui'
    metadata JSONB, -- Additional context data
    ip_address VARCHAR,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    session_token VARCHAR UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    ip_address VARCHAR,
    user_agent TEXT,
    device_info JSONB
);

-- API usage tracking
CREATE TABLE api_usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR,
    endpoint VARCHAR NOT NULL,
    method VARCHAR NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes (optimized for time-series queries)
CREATE INDEX idx_history_logs_user_time ON history_logs(user_id, created_at DESC);
CREATE INDEX idx_history_logs_action_type ON history_logs(action_type, created_at DESC);
CREATE INDEX idx_api_usage_logs_time ON api_usage_logs(created_at DESC);
```

### 8. Chat Service Database (`chat_db`)

```sql
-- Chat conversations
CREATE TABLE chat_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL, -- References auth_service.users.id
    title VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Chat messages
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id),
    user_id VARCHAR NOT NULL,
    message_type VARCHAR DEFAULT 'text', -- 'text', 'audio', 'system'
    content TEXT NOT NULL,
    metadata JSONB, -- For storing AI model info, confidence scores, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    is_user_message BOOLEAN DEFAULT TRUE -- FALSE for AI responses
);

-- Chat context (for maintaining conversation state)
CREATE TABLE chat_context (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id),
    context_data JSONB NOT NULL, -- Store conversation context for AI
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_chat_conversations_user ON chat_conversations(user_id, created_at DESC);
CREATE INDEX idx_chat_messages_conversation ON chat_messages(conversation_id, created_at);
```

### 9. Scheduler Service Database (`scheduler_db`)

```sql
-- Scheduled jobs
CREATE TABLE scheduled_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR NOT NULL, -- 'reminder_notification', 'cleanup', 'backup'
    job_data JSONB NOT NULL, -- Job parameters
    scheduled_at TIMESTAMP NOT NULL,
    status VARCHAR DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_attempt_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Job execution history
CREATE TABLE job_execution_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR NOT NULL,
    execution_time_ms INTEGER,
    output TEXT,
    error_message TEXT
);

-- Recurring job definitions
CREATE TABLE recurring_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR UNIQUE NOT NULL,
    job_type VARCHAR NOT NULL,
    cron_expression VARCHAR NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_scheduled_jobs_status_time ON scheduled_jobs(status, scheduled_at);
CREATE INDEX idx_recurring_jobs_next_run ON recurring_jobs(is_enabled, next_run_at);
```

---

## Project Structure

```
eindr-microservices/
├── services/
│   ├── api-gateway/
│   │   ├── kong.yml
│   │   └── docker-compose.yml
│   ├── auth-service/
│   │   ├── src/
│   │   │   ├── main.py
│   │   │   ├── models.py
│   │   │   ├── routers/
│   │   │   ├── services/
│   │   │   └── database.py
│   │   ├── migrations/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── user-service/
│   ├── reminder-service/
│   ├── note-service/
│   ├── ledger-service/
│   ├── friend-service/
│   ├── history-service/
│   ├── stt-service/
│   ├── tts-service/
│   ├── intent-service/
│   ├── chat-service/
│   └── scheduler-service/
├── shared/
│   ├── proto/
│   ├── events/
│   └── common/
├── infrastructure/
│   ├── docker-compose.yml
│   ├── kubernetes/
│   └── terraform/
└── docs/
    └── api/
```

---

## Migration Strategy

### Phase 1: Infrastructure Setup (Week 1-2)

1. Set up API Gateway (Kong)
2. Configure message bus (RabbitMQ)
3. Set up monitoring (Prometheus + Grafana)
4. Create shared libraries and protobuf definitions

### Phase 2: Core Services (Week 3-6)

1. Extract Auth Service
2. Extract User Service
3. Extract Reminder Service
4. Update API Gateway routing

### Phase 3: Domain Services (Week 7-10)

1. Extract Note Service
2. Extract Ledger Service
3. Extract Friend Service
4. Extract History Service

### Phase 4: AI Services (Week 11-13)

1. Extract STT Service
2. Extract TTS Service
3. Extract Intent Service
4. Extract Chat Service

### Phase 5: Background Services (Week 14)

1. Extract Scheduler Service
2. Implement event-driven communication
3. Performance testing and optimization

---

## Database Migration Scripts

Each service will need migration scripts to extract its data from the monolith database:

```sql
-- Example: Migrate users table to auth service
INSERT INTO auth_db.users (id, email, password_hash, created_at, is_active)
SELECT id, email, password_hash, created_at, true
FROM monolith_db.users;

-- Example: Migrate user preferences to user service
INSERT INTO user_db.user_profiles (user_id, language, timezone, created_at)
SELECT id, language, timezone, created_at
FROM monolith_db.users;

INSERT INTO user_db.user_preferences (user_id, allow_friends, receive_shared_notes, tts_language, chat_history_enabled)
SELECT user_id, allow_friends, receive_shared_notes, tts_language, chat_history_enabled
FROM monolith_db.preferences;
```

---

## Docker Compose for Local Development

```yaml
# docker-compose.yml
version: "3.8"

services:
  # Databases
  auth-db:
    image: postgres:15
    environment:
      POSTGRES_DB: auth_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"

  user-db:
    image: postgres:15
    environment:
      POSTGRES_DB: user_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"

  reminder-db:
    image: postgres:15
    environment:
      POSTGRES_DB: reminder_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5434:5432"

  # Message Bus
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"

  # Cache
  redis:
    image: redis:7
    ports:
      - "6379:6379"

  # API Gateway
  kong:
    image: kong:latest
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: /kong/declarative/kong.yml
    ports:
      - "8080:8000"
      - "8443:8443"
      - "8001:8001"
      - "8444:8444"

  # Microservices
  auth-service:
    build: ./services/auth-service
    environment:
      DATABASE_URL: postgresql://postgres:postgres@auth-db:5432/auth_db
    ports:
      - "8001:8000"
    depends_on:
      - auth-db

  user-service:
    build: ./services/user-service
    environment:
      DATABASE_URL: postgresql://postgres:postgres@user-db:5432/user_db
    ports:
      - "8002:8000"
    depends_on:
      - user-db

  # ... other services
```

This complete architecture provides clear separation of concerns, scalability, and maintainability while preserving all existing functionality of the monolith.
