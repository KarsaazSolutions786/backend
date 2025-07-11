from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    last_login = Column(DateTime, nullable=True)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    subscription_plan_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=True)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="customer", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversions"  # Note: table name is "conversions" not "conversations" in the database
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    title = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    last_message_at = Column(DateTime, nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversions_id = Column(Integer, ForeignKey("conversions.id"), nullable=False)  # Note: field name matches DB
    role = Column(String(50), nullable=True)  # 'user', 'assistant', 'system'
    description = Column(Text, nullable=True)  # Message content
    token_count = Column(Integer, nullable=True)
    model_used = Column(String(50), nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages") 