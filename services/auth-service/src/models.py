from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    last_login = Column(DateTime)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    subscription_plan_id = Column(Integer, ForeignKey("subscription_plans.id"))
    
    # Relationships
    sessions = relationship("CustomerSession", back_populates="customer", cascade="all, delete-orphan")
    login_attempt_logs = relationship("LoginAttempt", back_populates="customer", cascade="all, delete-orphan")
    subscription_plan = relationship("SubscriptionPlan", back_populates="customers")
    profile = relationship("CustomerProfile", back_populates="customer", uselist=False, cascade="all, delete-orphan")

class CustomerSession(Base):
    __tablename__ = "customer_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    session_token = Column(String(255), nullable=False)
    ip_address = Column(String(45))  # INET type maps to String in SQLAlchemy
    user_agent = Column(Text)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", back_populates="sessions")

class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    email = Column(String(255), nullable=False)
    ip_address = Column(String(45))  # INET type maps to String in SQLAlchemy
    user_agent = Column(String(255))
    is_success = Column(Boolean, default=False)  # Updated to match database schema
    failure_reason = Column(String(255))
    attempted_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", back_populates="login_attempt_logs")

class CustomerProfile(Base):
    __tablename__ = "customers_profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    full_name = Column(String(255))
    user_name = Column(String(255))
    bio = Column(Text)
    avatar_url = Column(String(500))
    phone_number = Column(String(20))
    date_of_birth = Column(DateTime)
    timezone_id = Column(Integer)
    language_id = Column(Integer)
    subscription_plan_id = Column(Integer)
    country = Column(String(100))
    city = Column(String(100))
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    gender = Column(String(50))
    
    # Relationships
    customer = relationship("Customer", back_populates="profile")

class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_name = Column(String(255), nullable=False)
    price = Column(DECIMAL(10, 2))
    billing_interval = Column(String(50))
    max_seats = Column(Integer)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    customers = relationship("Customer", back_populates="subscription_plan") 