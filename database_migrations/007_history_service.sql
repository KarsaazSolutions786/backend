-- History Service Database Schema
\c history_db;

-- Drop tables if they exist
DROP TABLE IF EXISTS api_usage_logs CASCADE;
DROP TABLE IF EXISTS login_history CASCADE;
DROP TABLE IF EXISTS activity_logs CASCADE;

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

-- Login history
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

-- Create indexes
CREATE INDEX idx_activity_logs_user_id ON activity_logs(user_id);
CREATE INDEX idx_activity_logs_created_at ON activity_logs(created_at);
CREATE INDEX idx_activity_logs_action ON activity_logs(action);
CREATE INDEX idx_activity_logs_resource ON activity_logs(resource_type, resource_id);
CREATE INDEX idx_login_history_user_id ON login_history(user_id);
CREATE INDEX idx_login_history_created_at ON login_history(created_at);
CREATE INDEX idx_api_usage_endpoint ON api_usage_logs(endpoint);
CREATE INDEX idx_api_usage_created_at ON api_usage_logs(created_at);
