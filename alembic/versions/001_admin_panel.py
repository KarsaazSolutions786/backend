"""Add admin panel tables

Revision ID: 001_admin_panel
Revises: 
Create Date: 2024-06-20 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '001_admin_panel'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Create admin panel tables with proper indexes and constraints."""
    
    # Create ENUM types
    admin_role_enum = postgresql.ENUM(
        'super_admin', 'support_agent', 'analyst', 'read_only',
        name='adminrole',
        create_type=True
    )
    
    notification_status_enum = postgresql.ENUM(
        'draft', 'scheduled', 'sent', 'failed', 'cancelled',
        name='notificationstatus',
        create_type=True
    )
    
    audit_action_enum = postgresql.ENUM(
        'create', 'update', 'delete', 'login', 'logout', 'view', 'export',
        'impersonate', 'feature_flag_toggle', 'user_deactivate', 'user_reactivate',
        'notification_send',
        name='auditaction',
        create_type=True
    )
    
    # Admin Users Table
    op.create_table(
        'admin_users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('role', admin_role_enum, nullable=False, default='read_only'),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        
        # 2FA Support
        sa.Column('two_fa_secret', sa.String(32), nullable=True),
        sa.Column('two_fa_enabled', sa.Boolean, default=False),
        sa.Column('backup_codes', postgresql.JSON, nullable=True),
        
        # Account Management
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('login_attempts', sa.Integer, default=0),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(timezone=True), default=sa.func.now()),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['created_by'], ['admin_users.id'], ondelete='SET NULL'),
    )
    
    # Admin Users Indexes
    op.create_index('idx_admin_users_id', 'admin_users', ['id'])
    op.create_index('idx_admin_users_email', 'admin_users', ['email'])
    op.create_index('idx_admin_email_active', 'admin_users', ['email', 'is_active'])
    op.create_index('idx_admin_role_active', 'admin_users', ['role', 'is_active'])
    op.create_index('idx_admin_last_login', 'admin_users', ['last_login_at'])
    
    # Admin Audit Log Table
    op.create_table(
        'admin_audit_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('admin_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('action', audit_action_enum, nullable=False),
        
        # Target Information
        sa.Column('target_type', sa.String(50), nullable=True),
        sa.Column('target_id', sa.String(255), nullable=True),
        
        # Action Details
        sa.Column('json_payload', postgresql.JSON, nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        
        # Request Context
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('request_id', sa.String(255), nullable=True),
        
        # Success/Failure
        sa.Column('success', sa.Boolean, default=True),
        sa.Column('error_message', sa.Text, nullable=True),
        
        # Timestamp
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['admin_id'], ['admin_users.id'], ondelete='CASCADE'),
    )
    
    # Audit Log Indexes
    op.create_index('idx_audit_log_id', 'admin_audit_log', ['id'])
    op.create_index('idx_audit_log_admin_id', 'admin_audit_log', ['admin_id'])
    op.create_index('idx_audit_log_action', 'admin_audit_log', ['action'])
    op.create_index('idx_audit_admin_action', 'admin_audit_log', ['admin_id', 'action'])
    op.create_index('idx_audit_target', 'admin_audit_log', ['target_type', 'target_id'])
    op.create_index('idx_audit_timestamp_desc', 'admin_audit_log', [sa.desc('timestamp')])
    op.create_index('idx_audit_ip_timestamp', 'admin_audit_log', ['ip_address', 'timestamp'])
    op.create_index('idx_audit_success', 'admin_audit_log', ['success'])
    op.create_index('idx_audit_request_id', 'admin_audit_log', ['request_id'])
    
    # Notifications Table
    op.create_table(
        'notifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        
        # Content
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('body', sa.Text, nullable=False),
        sa.Column('rich_content', postgresql.JSON, nullable=True),
        
        # Targeting
        sa.Column('audience_segment', sa.String(100), nullable=False),
        sa.Column('target_user_ids', postgresql.JSON, nullable=True),
        
        # Scheduling & Status
        sa.Column('status', notification_status_enum, nullable=False, default='draft'),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        
        # Delivery Stats
        sa.Column('total_recipients', sa.Integer, default=0),
        sa.Column('delivered_count', sa.Integer, default=0),
        sa.Column('opened_count', sa.Integer, default=0),
        sa.Column('clicked_count', sa.Integer, default=0),
        sa.Column('failed_count', sa.Integer, default=0),
        
        # Channels
        sa.Column('send_email', sa.Boolean, default=True),
        sa.Column('send_push', sa.Boolean, default=False),
        sa.Column('send_in_app', sa.Boolean, default=True),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=False),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['created_by'], ['admin_users.id'], ondelete='CASCADE'),
    )
    
    # Notifications Indexes
    op.create_index('idx_notifications_id', 'notifications', ['id'])
    op.create_index('idx_notification_status', 'notifications', ['status'])
    op.create_index('idx_notification_status_scheduled', 'notifications', ['status', 'scheduled_at'])
    op.create_index('idx_notification_audience', 'notifications', ['audience_segment', 'status'])
    op.create_index('idx_notification_created_by', 'notifications', ['created_by', 'created_at'])
    op.create_index('idx_notification_audience_segment', 'notifications', ['audience_segment'])
    
    # Feature Flags Table
    op.create_table(
        'feature_flags',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('key', sa.String(100), nullable=False, unique=True),
        
        # Configuration
        sa.Column('enabled', sa.Boolean, default=False),
        sa.Column('rollout_percentage', sa.Float, default=0.0),
        
        # Value & Metadata
        sa.Column('value', postgresql.JSON, nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('environment', sa.String(50), default='production'),
        
        # Targeting Rules
        sa.Column('target_user_segments', postgresql.JSON, nullable=True),
        sa.Column('target_user_ids', postgresql.JSON, nullable=True),
        
        # Lifecycle
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['created_by'], ['admin_users.id'], ondelete='CASCADE'),
    )
    
    # Feature Flags Indexes
    op.create_index('idx_feature_flags_id', 'feature_flags', ['id'])
    op.create_index('idx_feature_flags_key', 'feature_flags', ['key'])
    op.create_index('idx_feature_flag_enabled', 'feature_flags', ['enabled'])
    op.create_index('idx_feature_flag_enabled_env', 'feature_flags', ['enabled', 'environment'])
    op.create_index('idx_feature_flag_expires', 'feature_flags', ['expires_at'])
    op.create_index('idx_feature_flag_environment', 'feature_flags', ['environment'])
    
    # KPI Daily Snapshot Table
    op.create_table(
        'kpi_daily_snapshot',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False, unique=True),
        
        # User Metrics
        sa.Column('total_users', sa.Integer, default=0),
        sa.Column('new_users_today', sa.Integer, default=0),
        sa.Column('active_users_today', sa.Integer, default=0),
        sa.Column('trial_users', sa.Integer, default=0),
        sa.Column('premium_users', sa.Integer, default=0),
        sa.Column('churned_users_today', sa.Integer, default=0),
        
        # Content Metrics
        sa.Column('total_reminders', sa.Integer, default=0),
        sa.Column('reminders_created_today', sa.Integer, default=0),
        sa.Column('total_notes', sa.Integer, default=0),
        sa.Column('notes_created_today', sa.Integer, default=0),
        sa.Column('total_ledger_entries', sa.Integer, default=0),
        sa.Column('ledger_entries_today', sa.Integer, default=0),
        
        # AI Pipeline Metrics
        sa.Column('ai_requests_today', sa.Integer, default=0),
        sa.Column('ai_success_rate', sa.Float, default=0.0),
        sa.Column('avg_response_time_ms', sa.Float, default=0.0),
        
        # Engagement Metrics
        sa.Column('avg_session_duration_minutes', sa.Float, default=0.0),
        sa.Column('retention_rate_7d', sa.Float, default=0.0),
        sa.Column('retention_rate_30d', sa.Float, default=0.0),
        
        # Revenue Metrics
        sa.Column('revenue_today', sa.Float, default=0.0),
        sa.Column('mrr', sa.Float, default=0.0),
        
        # Computed at
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # KPI Daily Snapshot Indexes
    op.create_index('idx_kpi_daily_snapshot_id', 'kpi_daily_snapshot', ['id'])
    op.create_index('idx_kpi_date_desc', 'kpi_daily_snapshot', [sa.desc('date')])
    op.create_index('idx_kpi_computed_at', 'kpi_daily_snapshot', ['computed_at'])
    
    # Usage Hourly Table
    op.create_table(
        'usage_hourly',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('hour', sa.DateTime(timezone=True), nullable=False),
        
        # Request Metrics
        sa.Column('api_requests', sa.Integer, default=0),
        sa.Column('ai_pipeline_requests', sa.Integer, default=0),
        sa.Column('auth_requests', sa.Integer, default=0),
        
        # Performance Metrics
        sa.Column('avg_response_time_ms', sa.Float, default=0.0),
        sa.Column('error_rate', sa.Float, default=0.0),
        sa.Column('p95_response_time_ms', sa.Float, default=0.0),
        
        # Feature Usage
        sa.Column('voice_uploads', sa.Integer, default=0),
        sa.Column('text_interactions', sa.Integer, default=0),
        sa.Column('reminder_creations', sa.Integer, default=0),
        sa.Column('note_creations', sa.Integer, default=0),
        
        # Computed at
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Usage Hourly Indexes
    op.create_index('idx_usage_hourly_id', 'usage_hourly', ['id'])
    op.create_index('idx_usage_hour_desc', 'usage_hourly', [sa.desc('hour')])
    op.create_index('idx_usage_computed_at', 'usage_hourly', ['computed_at'])
    
    # Add constraints for data integrity
    op.create_check_constraint(
        'chk_rollout_percentage_range',
        'feature_flags',
        'rollout_percentage >= 0.0 AND rollout_percentage <= 100.0'
    )
    
    op.create_check_constraint(
        'chk_login_attempts_positive',
        'admin_users',
        'login_attempts >= 0'
    )


def downgrade() -> None:
    """Drop admin panel tables and ENUM types."""
    
    # Drop tables in reverse order to handle foreign key constraints
    op.drop_table('usage_hourly')
    op.drop_table('kpi_daily_snapshot')
    op.drop_table('feature_flags')
    op.drop_table('notifications')
    op.drop_table('admin_audit_log')
    op.drop_table('admin_users')
    
    # Drop ENUM types
    op.execute('DROP TYPE IF EXISTS auditaction')
    op.execute('DROP TYPE IF EXISTS notificationstatus')
    op.execute('DROP TYPE IF EXISTS adminrole') 