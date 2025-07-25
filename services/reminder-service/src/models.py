from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func, ForeignKey, Interval
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    reminders = relationship("Reminder", back_populates="customer", cascade="all, delete-orphan")
    shared_reminders = relationship("ReminderShare", foreign_keys="ReminderShare.shared_with_customer_id", back_populates="shared_with")
    shared_by_reminders = relationship("ReminderShare", foreign_keys="ReminderShare.owner_customer_id", back_populates="owner")
    reminder_notifications = relationship("ReminderNotification", back_populates="customer", cascade="all, delete-orphan")

class PriorityLevel(Base):
    __tablename__ = "priority_levels"
    
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(255), nullable=True)  # Matches actual schema: 'High', 'Medium', 'Low'
    rank = Column(Integer, nullable=True)  # 1, 2, 3
    
    # Relationships
    reminders = relationship("Reminder", back_populates="priority")

class Timezone(Base):
    __tablename__ = "timezones"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)  # 'UTC', etc.
    gmt_offset = Column(Interval, nullable=True)  # PostgreSQL interval type
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    reminders = relationship("Reminder", back_populates="timezone")

class RepeatPattern(Base):
    __tablename__ = "repeat_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    interval_value = Column(Integer, nullable=False)  # e.g. 1, 2, 3
    interval_unit = Column(String(50), nullable=False)  # e.g. day, week, month, year
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    reminders = relationship("Reminder", back_populates="repeat_pattern")

class Reminder(Base):
    __tablename__ = "reminders"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    title = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    time = Column(DateTime, nullable=True)
    repeat_pattern_id = Column(Integer, ForeignKey("repeat_patterns.id"), nullable=True)
    timezone_id = Column(Integer, ForeignKey("timezones.id"), nullable=True)
    is_shared = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    next_occurrence = Column(DateTime, nullable=True)
    occurrence_count = Column(Integer, default=0)
    max_occurrence = Column(Integer, nullable=True)  # Max times to repeat
    priority_id = Column(Integer, ForeignKey("priority_levels.id"), nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    is_completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="reminders")
    priority = relationship("PriorityLevel", back_populates="reminders")
    timezone = relationship("Timezone", back_populates="reminders")
    repeat_pattern = relationship("RepeatPattern", back_populates="reminders")
    shares = relationship("ReminderShare", back_populates="reminder", cascade="all, delete-orphan")
    notifications = relationship("ReminderNotification", back_populates="reminder", cascade="all, delete-orphan")

class ReminderShare(Base):
    __tablename__ = "reminder_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    reminder_id = Column(Integer, ForeignKey("reminders.id"), nullable=False)
    owner_customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    shared_with_customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    can_edit = Column(Boolean, default=False)
    can_complete = Column(Boolean, default=True)
    can_reshedule = Column(Boolean, default=True)
    status = Column(String(50), nullable=True)
    shared_at = Column(DateTime, default=func.current_timestamp())
    responded_at = Column(DateTime, nullable=True)
    
    # Relationships
    reminder = relationship("Reminder", back_populates="shares")
    owner = relationship("Customer", foreign_keys=[owner_customer_id], back_populates="shared_by_reminders")
    shared_with = relationship("Customer", foreign_keys=[shared_with_customer_id], back_populates="shared_reminders")

class ReminderNotification(Base):
    __tablename__ = "reminder_notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    reminder_id = Column(Integer, ForeignKey("reminders.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    notification_type = Column(String, nullable=False)  # 'email', 'push', 'sms'
    scheduled_at = Column(DateTime, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    delivery_status = Column(String, default="pending")  # 'pending', 'sent', 'delivered', 'failed'
    failure_reason = Column(String, nullable=True)
    notification_content = Column(Text, nullable=True)  # JSON string with title, body, etc.
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    reminder = relationship("Reminder", back_populates="notifications")
    customer = relationship("Customer", back_populates="reminder_notifications")