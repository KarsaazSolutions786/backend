-- ===============================================
-- EINDR MICROSERVICES DATABASE SETUP
-- ===============================================

-- ===============================================
-- 1. AUTH SERVICE DATABASE (Port 5442)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS auth_db;
CREATE DATABASE auth_db;
\c auth_db;

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    password_reset_token VARCHAR(255),
    password_reset_expires TIMESTAMP,
    verification_token VARCHAR(255),
    verification_expires TIMESTAMP
);

-- Refresh tokens for JWT
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    is_revoked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for auth_db
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_verification_token ON users(verification_token);
CREATE INDEX idx_users_reset_token ON users(password_reset_token);
CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_expires ON refresh_tokens(expires_at);
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_token ON user_sessions(session_token);

-- ===============================================
-- 2. USER SERVICE DATABASE (Port 5433)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS user_db;
CREATE DATABASE user_db;
\c user_db;

-- User profiles
CREATE TABLE user_profiles (
    user_id VARCHAR PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    display_name VARCHAR(150),
    bio TEXT,
    avatar_url VARCHAR(500),
    phone_number VARCHAR(20),
    date_of_birth DATE,
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    country VARCHAR(3),
    city VARCHAR(100),
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
    notification_frequency VARCHAR DEFAULT 'immediate',
    auto_backup BOOLEAN DEFAULT TRUE,
    data_retention_days INTEGER DEFAULT 365,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User devices for push notifications
CREATE TABLE user_devices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL REFERENCES user_profiles(user_id),
    device_token VARCHAR NOT NULL,
    device_type VARCHAR NOT NULL,
    device_name VARCHAR,
    device_model VARCHAR,
    os_version VARCHAR,
    app_version VARCHAR,
    is_active BOOLEAN DEFAULT TRUE,
    last_seen TIMESTAMP DEFAULT NOW(),
    registered_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for user_db
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_devices_user_id ON user_devices(user_id);
CREATE INDEX idx_user_devices_token ON user_devices(device_token);

-- ===============================================
-- 3. REMINDER SERVICE DATABASE (Port 5434)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS reminder_db;
CREATE DATABASE reminder_db;
\c reminder_db;

-- Main reminders table
CREATE TABLE reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    time TIMESTAMP NOT NULL,
    repeat_pattern VARCHAR DEFAULT 'none',
    timezone VARCHAR DEFAULT 'UTC',
    is_shared BOOLEAN DEFAULT FALSE,
    created_by VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    next_occurrence TIMESTAMP,
    occurrence_count INTEGER DEFAULT 0,
    max_occurrences INTEGER,
    priority VARCHAR DEFAULT 'medium',
    category VARCHAR,
    tags TEXT[],
    location_name VARCHAR,
    attachment_urls TEXT[]
);

-- Reminder notifications tracking
CREATE TABLE reminder_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id UUID NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
    user_id VARCHAR NOT NULL,
    notification_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
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
    reminder_id UUID NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
    owner_user_id VARCHAR NOT NULL,
    shared_with_user_id VARCHAR NOT NULL,
    can_edit BOOLEAN DEFAULT FALSE,
    can_complete BOOLEAN DEFAULT TRUE,
    can_reschedule BOOLEAN DEFAULT FALSE,
    status VARCHAR DEFAULT 'pending',
    shared_at TIMESTAMP DEFAULT NOW(),
    responded_at TIMESTAMP
);

-- Indexes for reminder_db
CREATE INDEX idx_reminders_user_time ON reminders(user_id, time);
CREATE INDEX idx_reminders_next_occurrence ON reminders(next_occurrence);
CREATE INDEX idx_reminders_priority ON reminders(priority, user_id);
CREATE INDEX idx_reminders_category ON reminders(category, user_id);
CREATE INDEX idx_reminder_notifications_pending ON reminder_notifications(status, scheduled_at);
CREATE INDEX idx_reminder_shares_user ON reminder_shares(shared_with_user_id);

-- ===============================================
-- 4. NOTE SERVICE DATABASE (Port 5435)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS note_db;
CREATE DATABASE note_db;
\c note_db;

-- Folders for organizing notes
CREATE TABLE note_folders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    color VARCHAR DEFAULT '#1f77b4',
    parent_folder_id UUID REFERENCES note_folders(id),
    is_shared BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Notes table
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    folder_id UUID REFERENCES note_folders(id),
    title VARCHAR NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR DEFAULT 'text',
    tags TEXT[],
    category VARCHAR,
    is_shared BOOLEAN DEFAULT FALSE,
    is_favorite BOOLEAN DEFAULT FALSE,
    is_pinned BOOLEAN DEFAULT FALSE,
    word_count INTEGER DEFAULT 0,
    character_count INTEGER DEFAULT 0,
    reading_time INTEGER DEFAULT 0,
    attachment_urls TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Note sharing
CREATE TABLE note_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    note_id UUID NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    owner_user_id VARCHAR NOT NULL,
    shared_with_user_id VARCHAR NOT NULL,
    can_edit BOOLEAN DEFAULT FALSE,
    can_comment BOOLEAN DEFAULT TRUE,
    status VARCHAR DEFAULT 'pending',
    shared_at TIMESTAMP DEFAULT NOW(),
    responded_at TIMESTAMP
);

-- Indexes for note_db
CREATE INDEX idx_notes_user_id ON notes(user_id);
CREATE INDEX idx_notes_folder_id ON notes(folder_id);
CREATE INDEX idx_notes_created_at ON notes(created_at);
CREATE INDEX idx_notes_updated_at ON notes(updated_at);
CREATE INDEX idx_note_folders_user_id ON note_folders(user_id);
CREATE INDEX idx_note_shares_note_id ON note_shares(note_id);

-- ===============================================
-- 5. LEDGER SERVICE DATABASE (Port 5436)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS ledger_db;
CREATE DATABASE ledger_db;
\c ledger_db;

-- Expense categories
CREATE TABLE expense_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    color VARCHAR DEFAULT '#1f77b4',
    icon VARCHAR,
    is_default BOOLEAN DEFAULT FALSE,
    budget_limit DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Main expenses table
CREATE TABLE expenses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    category_id UUID REFERENCES expense_categories(id),
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    description TEXT NOT NULL,
    date DATE NOT NULL,
    payment_method VARCHAR,
    payment_account VARCHAR,
    tags TEXT[],
    location_name VARCHAR,
    receipt_urls TEXT[],
    is_recurring BOOLEAN DEFAULT FALSE,
    recurring_frequency VARCHAR,
    recurring_end_date DATE,
    notes TEXT,
    tax_amount DECIMAL(10,2) DEFAULT 0,
    tax_rate DECIMAL(5,2) DEFAULT 0,
    is_business_expense BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Budgets
CREATE TABLE budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    category_id UUID REFERENCES expense_categories(id),
    name VARCHAR NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    period VARCHAR NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    alert_threshold DECIMAL(5,2) DEFAULT 80.0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for ledger_db
CREATE INDEX idx_expenses_user_id ON expenses(user_id);
CREATE INDEX idx_expenses_date ON expenses(date);
CREATE INDEX idx_expenses_category ON expenses(category_id);
CREATE INDEX idx_expenses_amount ON expenses(amount);
CREATE INDEX idx_budgets_user_id ON budgets(user_id);
CREATE INDEX idx_budgets_period ON budgets(start_date, end_date);

-- ===============================================
-- 6. FRIEND SERVICE DATABASE (Port 5437)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS friend_db;
CREATE DATABASE friend_db;
\c friend_db;

-- Friendships table
CREATE TABLE friendships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    friend_id VARCHAR NOT NULL,
    status VARCHAR CHECK (status IN ('pending', 'accepted', 'blocked', 'declined')) DEFAULT 'pending',
    initiated_by VARCHAR NOT NULL,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    accepted_at TIMESTAMP,
    UNIQUE(user_id, friend_id),
    CHECK (user_id != friend_id)
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
    can_view_activity BOOLEAN DEFAULT FALSE,
    notification_level VARCHAR DEFAULT 'normal',
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, friend_id)
);

-- Friend request history
CREATE TABLE friend_request_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requester_id VARCHAR NOT NULL,
    requested_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    message TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for friend_db
CREATE INDEX idx_friendships_user_id ON friendships(user_id);
CREATE INDEX idx_friendships_friend_id ON friendships(friend_id);
CREATE INDEX idx_friendships_status ON friendships(status);
CREATE INDEX idx_friend_permissions_user_friend ON friend_permissions(user_id, friend_id);

-- ===============================================
-- 7. HISTORY SERVICE DATABASE (Port 5438)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS history_db;
CREATE DATABASE history_db;
\c history_db;

-- Activity logs
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    session_id VARCHAR,
    action VARCHAR NOT NULL,
    resource_type VARCHAR NOT NULL,
    resource_id VARCHAR,
    old_values JSONB,
    new_values JSONB,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    location_country VARCHAR(3),
    location_city VARCHAR(100),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User login history
CREATE TABLE login_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    login_type VARCHAR NOT NULL,
    ip_address INET,
    user_agent TEXT,
    device_type VARCHAR,
    location_country VARCHAR(3),
    location_city VARCHAR(100),
    success BOOLEAN DEFAULT TRUE,
    failure_reason VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);

-- API usage tracking
CREATE TABLE api_usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR,
    endpoint VARCHAR NOT NULL,
    method VARCHAR NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for history_db
CREATE INDEX idx_activity_logs_user_id ON activity_logs(user_id);
CREATE INDEX idx_activity_logs_created_at ON activity_logs(created_at);
CREATE INDEX idx_activity_logs_action ON activity_logs(action);
CREATE INDEX idx_activity_logs_resource ON activity_logs(resource_type, resource_id);
CREATE INDEX idx_login_history_user_id ON login_history(user_id);
CREATE INDEX idx_login_history_created_at ON login_history(created_at);
CREATE INDEX idx_api_usage_endpoint ON api_usage_logs(endpoint);
CREATE INDEX idx_api_usage_created_at ON api_usage_logs(created_at);

-- ===============================================
-- 8. CHAT SERVICE DATABASE (Port 5439)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS chat_db;
CREATE DATABASE chat_db;
\c chat_db;

-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL,
    title VARCHAR,
    context JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_message_at TIMESTAMP DEFAULT NOW()
);

-- Chat messages
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB,
    token_count INTEGER,
    model_used VARCHAR,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for chat_db
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at);
CREATE INDEX idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);

-- ===============================================
-- 9. KONG DATABASE (Port 5441)
-- ===============================================
\c postgres;
DROP DATABASE IF EXISTS kong_db;
CREATE DATABASE kong_db;

-- ===============================================
-- DATABASE SETUP COMPLETE
-- ===============================================
