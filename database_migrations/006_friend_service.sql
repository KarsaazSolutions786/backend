-- Friend Service Database Schema
\c friend_db;

-- Drop tables if they exist
DROP TABLE IF EXISTS friend_request_history CASCADE;
DROP TABLE IF EXISTS friend_permissions CASCADE;
DROP TABLE IF EXISTS friendships CASCADE;

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

-- Create indexes
CREATE INDEX idx_friendships_user_id ON friendships(user_id);
CREATE INDEX idx_friendships_friend_id ON friendships(friend_id);
CREATE INDEX idx_friendships_status ON friendships(status);
CREATE INDEX idx_friend_permissions_user_friend ON friend_permissions(user_id, friend_id);
