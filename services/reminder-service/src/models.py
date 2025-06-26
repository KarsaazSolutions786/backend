from sqlalchemy import Column, String, Text, Boolean, TIMESTAMP, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
from datetime import datetime
from .database import Base
import uuid

class Reminder(Base):
    """Reminder model - stores all reminder data"""
    __tablename__ = "reminders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)  # Reference to auth service user
    title = Column(Text)
    description = Column(Text)
    time = Column(TIMESTAMP)
    repeat_pattern = Column(String)  # 'none', 'daily', 'weekly', 'monthly', 'yearly'
    timezone = Column(String)
    is_shared = Column(Boolean, default=False)
    created_by = Column(String)  # Reference to auth service user who created (for shared reminders)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Reminder state
    is_completed = Column(Boolean, default=False)
    completed_at = Column(TIMESTAMP, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Recurrence tracking
    next_occurrence = Column(TIMESTAMP, nullable=True)
    occurrence_count = Column(String, default="0")
    max_occurrences = Column(String, nullable=True)
    
    # Metadata
    priority = Column(String, default="medium")  # 'low', 'medium', 'high', 'urgent'
    category = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    
    __table_args__ = (
        Index('idx_reminders_user_time', 'user_id', 'time'),
        Index('idx_reminders_user_active', 'user_id', 'is_active'),
        Index('idx_reminders_next_occurrence', 'next_occurrence'),
        Index('idx_reminders_shared', 'is_shared', 'user_id'),
        Index('idx_reminders_created_by', 'created_by'),
        Index('idx_reminders_priority', 'priority', 'user_id'),
    )

class ReminderNotification(Base):
    """Track reminder notification delivery"""
    __tablename__ = "reminder_notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    reminder_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    
    # Notification details
    notification_type = Column(String, nullable=False)  # 'push', 'email', 'sms'
    status = Column(String, nullable=False)  # 'pending', 'sent', 'delivered', 'failed'
    scheduled_at = Column(TIMESTAMP, nullable=False)
    sent_at = Column(TIMESTAMP, nullable=True)
    delivered_at = Column(TIMESTAMP, nullable=True)
    
    # Delivery tracking
    delivery_attempts = Column(String, default="0")
    last_attempt_at = Column(TIMESTAMP, nullable=True)
    failure_reason = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_reminder_notifications_reminder', 'reminder_id'),
        Index('idx_reminder_notifications_user_status', 'user_id', 'status'),
        Index('idx_reminder_notifications_scheduled', 'scheduled_at'),
        Index('idx_reminder_notifications_pending', 'status', 'scheduled_at'),
    )

class ReminderShare(Base):
    """Track reminder sharing between users"""
    __tablename__ = "reminder_shares"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    reminder_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    owner_user_id = Column(String, nullable=False)  # Original reminder owner
    shared_with_user_id = Column(String, nullable=False)  # User reminder is shared with
    
    # Share permissions
    can_edit = Column(Boolean, default=False)
    can_complete = Column(Boolean, default=True)
    can_reschedule = Column(Boolean, default=False)
    
    # Share status
    status = Column(String, default="pending")  # 'pending', 'accepted', 'declined'
    shared_at = Column(TIMESTAMP, default=datetime.utcnow)
    responded_at = Column(TIMESTAMP, nullable=True)
    
    __table_args__ = (
        Index('idx_reminder_shares_reminder', 'reminder_id'),
        Index('idx_reminder_shares_owner', 'owner_user_id'),
        Index('idx_reminder_shares_shared_with', 'shared_with_user_id'),
        Index('idx_reminder_shares_status', 'status'),
    ) 