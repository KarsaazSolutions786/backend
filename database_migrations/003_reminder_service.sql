-- Reminder Service Database Schema
\c reminder_db;

-- Drop tables if they exist
DROP TABLE IF EXISTS reminder_shares CASCADE;
DROP TABLE IF EXISTS reminder_notifications CASCADE;
DROP TABLE IF EXISTS reminders CASCADE;

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

-- Reminder notifications
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

-- Create indexes
CREATE INDEX idx_reminders_user_time ON reminders(user_id, time);
CREATE INDEX idx_reminders_next_occurrence ON reminders(next_occurrence);
CREATE INDEX idx_reminders_priority ON reminders(priority, user_id);
CREATE INDEX idx_reminders_category ON reminders(category, user_id);
CREATE INDEX idx_reminder_notifications_pending ON reminder_notifications(status, scheduled_at);
CREATE INDEX idx_reminder_shares_user ON reminder_shares(shared_with_user_id);
