from sqlalchemy import Column, String, Boolean, TIMESTAMP, Index
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from .database import Base
import uuid

class User(Base):
    """User authentication model - contains only auth-related data"""
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=True)  # Optional for Firebase users
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(TIMESTAMP, nullable=True)
    
    # Authentication metadata
    login_attempts = Column(String, default="0")
    locked_until = Column(TIMESTAMP, nullable=True)
    password_reset_token = Column(String, nullable=True)
    password_reset_expires = Column(TIMESTAMP, nullable=True)
    email_verification_token = Column(String, nullable=True)
    email_verification_expires = Column(TIMESTAMP, nullable=True)
    
    # Device/session tracking
    last_login_ip = Column(String, nullable=True)
    last_user_agent = Column(String, nullable=True)
    
    __table_args__ = (
        Index('idx_users_email_active', 'email', 'is_active'),
        Index('idx_users_created_at', 'created_at'),
        Index('idx_users_verification_token', 'email_verification_token'),
        Index('idx_users_reset_token', 'password_reset_token'),
    )

class RefreshToken(Base):
    """Refresh token storage for JWT authentication"""
    __tablename__ = "refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    token_hash = Column(String, nullable=False, unique=True)
    expires_at = Column(TIMESTAMP, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    revoked_at = Column(TIMESTAMP, nullable=True)
    device_info = Column(String, nullable=True)  # User agent or device identifier
    
    __table_args__ = (
        Index('idx_refresh_tokens_user', 'user_id'),
        Index('idx_refresh_tokens_expires', 'expires_at'),
        Index('idx_refresh_tokens_hash', 'token_hash'),
    )

class LoginAttempt(Base):
    """Track login attempts for security monitoring"""
    __tablename__ = "login_attempts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, nullable=False, index=True)
    ip_address = Column(String, nullable=False)
    user_agent = Column(String, nullable=True)
    success = Column(Boolean, nullable=False)
    failure_reason = Column(String, nullable=True)  # 'invalid_password', 'user_not_found', etc.
    attempted_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_login_attempts_email_time', 'email', 'attempted_at'),
        Index('idx_login_attempts_ip_time', 'ip_address', 'attempted_at'),
    ) 