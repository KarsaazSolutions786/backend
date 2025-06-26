-- User Service Database Schema
\c user_db;

-- Drop tables if they exist
DROP TABLE IF EXISTS user_devices CASCADE;
DROP TABLE IF EXISTS user_preferences CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;

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

-- User devices
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

-- Create indexes
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_devices_user_id ON user_devices(user_id);
CREATE INDEX idx_user_devices_token ON user_devices(device_token);
