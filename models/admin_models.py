"""
Admin Panel Database Models
============================

Comprehensive models for admin panel functionality including:
- Admin user management with RBAC
- Audit logging for all admin actions  
- Notification system for user broadcasts
- Feature flags for rollout control
- KPI materialized views for performance
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index, event
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from enum import Enum
import bcrypt

Base = declarative_base()

# Enums for type safety
class AdminRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    SUPPORT_AGENT = "support_agent"  
    ANALYST = "analyst"
    READ_ONLY = "read_only"

class NotificationStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AuditAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    VIEW = "view"
    EXPORT = "export"
    IMPERSONATE = "impersonate"
    FEATURE_FLAG_TOGGLE = "feature_flag_toggle"
    USER_DEACTIVATE = "user_deactivate"
    USER_REACTIVATE = "user_reactivate"
    NOTIFICATION_SEND = "notification_send"

# Admin Users Table
class AdminUser(Base):
    __tablename__ = "admin_users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(ENUM(AdminRole, values_callable=lambda obj: [e.value for e in obj]), nullable=False, default=AdminRole.READ_ONLY)
    hashed_password = Column(String(255), nullable=False)
    
    # 2FA Support
    two_fa_secret = Column(String(32), nullable=True)  # Base32 encoded secret
    two_fa_enabled = Column(Boolean, default=False)
    backup_codes = Column(JSON, nullable=True)  # List of one-time backup codes
    
    # Account Management
    is_active = Column(Boolean, default=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), default=func.now())
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("admin_users.id"), nullable=True)
    
    # Relationships
    audit_logs = relationship("AdminAuditLog", back_populates="admin_user")
    notifications_created = relationship("Notification", back_populates="created_by_user")
    feature_flags_created = relationship("FeatureFlag", back_populates="created_by_user")
    
    # Indexes
    __table_args__ = (
        Index('idx_admin_email_active', 'email', 'is_active'),
        Index('idx_admin_role_active', 'role', 'is_active'),
        Index('idx_admin_last_login', 'last_login_at'),
    )
    
    def set_password(self, password: str) -> None:
        """Hash and set password with bcrypt."""
        salt = bcrypt.gensalt()
        self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        self.password_changed_at = func.now()
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))
    
    def has_permission(self, required_roles: list[AdminRole]) -> bool:
        """Check if user has any of the required roles."""
        return self.is_active and self.role in required_roles
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': str(self.id),
            'name': self.name,
            'email': self.email,
            'role': self.role.value,
            'two_fa_enabled': self.two_fa_enabled,
            'is_active': self.is_active,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Audit Log Table
class AdminAuditLog(Base):
    __tablename__ = "admin_audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    admin_id = Column(UUID(as_uuid=True), ForeignKey("admin_users.id"), nullable=False, index=True)
    action = Column(ENUM(AuditAction), nullable=False, index=True)
    
    # Target Information
    target_type = Column(String(50), nullable=True, index=True)  # 'user', 'reminder', 'feature_flag', etc.
    target_id = Column(String(255), nullable=True, index=True)   # ID of the affected resource
    
    # Action Details
    json_payload = Column(JSON, nullable=True)  # Before/after state, additional context
    description = Column(Text, nullable=True)   # Human-readable description
    
    # Request Context
    ip_address = Column(String(45), nullable=True, index=True)  # IPv4/IPv6 support
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(255), nullable=True, index=True)  # For request correlation
    
    # Success/Failure
    success = Column(Boolean, default=True, index=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    admin_user = relationship("AdminUser", back_populates="audit_logs")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_audit_admin_action', 'admin_id', 'action'),
        Index('idx_audit_target', 'target_type', 'target_id'),
        Index('idx_audit_timestamp_desc', 'timestamp'),
        Index('idx_audit_ip_timestamp', 'ip_address', 'timestamp'),
    )

# Notifications Table
class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Content
    title = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)
    rich_content = Column(JSON, nullable=True)  # HTML, markdown, attachments
    
    # Targeting
    audience_segment = Column(String(100), nullable=False, index=True)  # 'all', 'trial_users', 'premium', 'inactive'
    target_user_ids = Column(JSON, nullable=True)  # Specific user IDs for targeted sends
    
    # Scheduling & Status
    status = Column(ENUM(NotificationStatus), nullable=False, default=NotificationStatus.DRAFT, index=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True, index=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    
    # Delivery Stats
    total_recipients = Column(Integer, default=0)
    delivered_count = Column(Integer, default=0)
    opened_count = Column(Integer, default=0)
    clicked_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    
    # Channels
    send_email = Column(Boolean, default=True)
    send_push = Column(Boolean, default=False)
    send_in_app = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("admin_users.id"), nullable=False, index=True)
    
    # Relationships
    created_by_user = relationship("AdminUser", back_populates="notifications_created")
    
    # Indexes
    __table_args__ = (
        Index('idx_notification_status_scheduled', 'status', 'scheduled_at'),
        Index('idx_notification_audience', 'audience_segment', 'status'),
        Index('idx_notification_created_by', 'created_by', 'created_at'),
    )

# Feature Flags Table
class FeatureFlag(Base):
    __tablename__ = "feature_flags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)  # e.g., 'ai_pipeline_v2'
    
    # Configuration
    enabled = Column(Boolean, default=False, index=True)
    rollout_percentage = Column(Float, default=0.0)  # 0.0 to 100.0
    
    # Value & Metadata
    value = Column(JSON, nullable=True)  # For complex feature configurations
    description = Column(Text, nullable=True)
    environment = Column(String(50), default='production', index=True)  # 'dev', 'staging', 'production'
    
    # Targeting Rules
    target_user_segments = Column(JSON, nullable=True)  # User segments to include/exclude
    target_user_ids = Column(JSON, nullable=True)       # Specific user IDs
    
    # Lifecycle
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(UUID(as_uuid=True), ForeignKey("admin_users.id"), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Relationships
    created_by_user = relationship("AdminUser", back_populates="feature_flags_created")
    
    # Indexes
    __table_args__ = (
        Index('idx_feature_flag_enabled_env', 'enabled', 'environment'),
        Index('idx_feature_flag_expires', 'expires_at'),
    )

# KPI Daily Snapshot (Materialized View)
class KpiDailySnapshot(Base):
    __tablename__ = "kpi_daily_snapshot"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime(timezone=True), nullable=False, unique=True, index=True)
    
    # User Metrics
    total_users = Column(Integer, default=0)
    new_users_today = Column(Integer, default=0)
    active_users_today = Column(Integer, default=0)
    trial_users = Column(Integer, default=0)
    premium_users = Column(Integer, default=0)
    churned_users_today = Column(Integer, default=0)
    
    # Content Metrics
    total_reminders = Column(Integer, default=0)
    reminders_created_today = Column(Integer, default=0)
    total_notes = Column(Integer, default=0)
    notes_created_today = Column(Integer, default=0)
    total_ledger_entries = Column(Integer, default=0)
    ledger_entries_today = Column(Integer, default=0)
    
    # AI Pipeline Metrics
    ai_requests_today = Column(Integer, default=0)
    ai_success_rate = Column(Float, default=0.0)
    avg_response_time_ms = Column(Float, default=0.0)
    
    # Engagement Metrics
    avg_session_duration_minutes = Column(Float, default=0.0)
    retention_rate_7d = Column(Float, default=0.0)
    retention_rate_30d = Column(Float, default=0.0)
    
    # Revenue Metrics (if applicable)
    revenue_today = Column(Float, default=0.0)
    mrr = Column(Float, default=0.0)  # Monthly Recurring Revenue
    
    # Computed at
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_kpi_date_desc', 'date'),
    )

# Usage Hourly Table for granular analytics
class UsageHourly(Base):
    __tablename__ = "usage_hourly"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hour = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Request Metrics
    api_requests = Column(Integer, default=0)
    ai_pipeline_requests = Column(Integer, default=0)
    auth_requests = Column(Integer, default=0)
    
    # Performance Metrics
    avg_response_time_ms = Column(Float, default=0.0)
    error_rate = Column(Float, default=0.0)
    p95_response_time_ms = Column(Float, default=0.0)
    
    # Feature Usage
    voice_uploads = Column(Integer, default=0)
    text_interactions = Column(Integer, default=0)
    reminder_creations = Column(Integer, default=0)
    note_creations = Column(Integer, default=0)
    
    # Computed at
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_usage_hour_desc', 'hour'),
    )

# SQLAlchemy event listeners for automatic audit logging
@event.listens_for(AdminUser, 'after_insert')
def log_admin_user_creation(mapper, connection, target):
    """Automatically log admin user creation."""
    # This will be handled by the audit utility in practice
    pass

@event.listens_for(FeatureFlag, 'after_update')
def log_feature_flag_update(mapper, connection, target):
    """Automatically log feature flag changes."""
    # This will be handled by the audit utility in practice
    pass 