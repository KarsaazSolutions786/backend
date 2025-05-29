from sqlalchemy import Column, String, DateTime, Boolean, Float, ForeignKey, Text, Integer, TIMESTAMP, CheckConstraint, ARRAY, NUMERIC
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
from connect_db import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=True)  # Optional for Firebase users
    language = Column(String)
    timezone = Column(String)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    preferences = relationship("Preferences", back_populates="user", uselist=False, cascade="all, delete-orphan")
    reminders = relationship("Reminder", foreign_keys="Reminder.user_id", back_populates="user", cascade="all, delete-orphan")
    created_reminders = relationship("Reminder", foreign_keys="Reminder.created_by", back_populates="creator")
    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")
    ledger_entries = relationship("LedgerEntry", back_populates="user", cascade="all, delete-orphan")
    friendships = relationship("Friendship", foreign_keys="Friendship.user_id", back_populates="user", cascade="all, delete-orphan")
    friend_relationships = relationship("Friendship", foreign_keys="Friendship.friend_id", back_populates="friend")
    permissions_given = relationship("Permission", foreign_keys="Permission.user_id", back_populates="user", cascade="all, delete-orphan")
    permissions_received = relationship("Permission", foreign_keys="Permission.friend_id", back_populates="friend")
    embeddings = relationship("Embedding", back_populates="user", cascade="all, delete-orphan")
    history_logs = relationship("HistoryLog", back_populates="user", cascade="all, delete-orphan")

class Preferences(Base):
    __tablename__ = "preferences"

    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    allow_friends = Column(Boolean)
    receive_shared_notes = Column(Boolean)
    notification_sound = Column(String)
    tts_language = Column(String)
    chat_history_enabled = Column(Boolean)

    # Relationships
    user = relationship("User", back_populates="preferences")

class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(Text)
    description = Column(Text)
    time = Column(TIMESTAMP)
    repeat_pattern = Column(String)
    timezone = Column(String)
    is_shared = Column(Boolean)
    created_by = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="reminders")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_reminders")
    embeddings = relationship("Embedding", back_populates="reminder", cascade="all, delete-orphan")

class Note(Base):
    __tablename__ = "notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text)
    source = Column(String)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="notes")

class LedgerEntry(Base):
    __tablename__ = "ledger_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    contact_name = Column(String)
    amount = Column(NUMERIC)
    direction = Column(String, CheckConstraint("direction IN ('owe', 'owed')"))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="ledger_entries")

class Friendship(Base):
    __tablename__ = "friendships"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    friend_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, CheckConstraint("status IN ('pending', 'accepted', 'blocked')"))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="friendships")
    friend = relationship("User", foreign_keys=[friend_id], back_populates="friend_relationships")

class Permission(Base):
    __tablename__ = "permissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    friend_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    auto_accept_reminders = Column(Boolean)
    auto_accept_notes = Column(Boolean)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="permissions_given")
    friend = relationship("User", foreign_keys=[friend_id], back_populates="permissions_received")

class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    reminder_id = Column(UUID(as_uuid=True), ForeignKey("reminders.id", ondelete="CASCADE"))
    embedding = Column(ARRAY(Float))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="embeddings")
    reminder = relationship("Reminder", back_populates="embeddings")

class HistoryLog(Base):
    __tablename__ = "history_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text)
    interaction_type = Column(String)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="history_logs") 