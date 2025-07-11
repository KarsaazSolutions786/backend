from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func, ForeignKey, Numeric
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
    subscriptions = relationship("CustomerSubscription", back_populates="customer", cascade="all, delete-orphan")
    subscription_history = relationship("CustomerSubscriptionHistory", back_populates="customer", cascade="all, delete-orphan")

class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    plan_name = Column(String(255), nullable=False)
    price = Column(Numeric(10, 2), nullable=True)
    billing_interval = Column(String(50), nullable=True)
    max_seats = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    customers = relationship("Customer")
    customer_subscriptions = relationship("CustomerSubscription", back_populates="subscription_plan")
    subscription_history = relationship("CustomerSubscriptionHistory", back_populates="subscription_plan")

class RepeatPattern(Base):
    __tablename__ = "repeat_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    interval_value = Column(Integer, nullable=False)  # e.g. 1, 2, 3
    interval_unit = Column(String(50), nullable=False)  # e.g. day, week, month, year
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customer_subscriptions = relationship("CustomerSubscription", back_populates="repeat_pattern")
    subscription_history = relationship("CustomerSubscriptionHistory", back_populates="repeat_pattern")

class CustomerSubscription(Base):
    __tablename__ = "customer_subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    subscription_plan_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=False)
    subscription_start_date = Column(DateTime, nullable=False)
    subscription_end_date = Column(DateTime, nullable=True)
    repeat_pattern_id = Column(Integer, ForeignKey("repeat_patterns.id"), nullable=True)
    status = Column(String(50), nullable=False, default='active')  # active, cancelled, expired, suspended
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", back_populates="subscriptions")
    subscription_plan = relationship("SubscriptionPlan", back_populates="customer_subscriptions")
    repeat_pattern = relationship("RepeatPattern", back_populates="customer_subscriptions")

class CustomerSubscriptionHistory(Base):
    __tablename__ = "customer_subscription_history"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    subscription_plan_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=False)
    subscription_start_date = Column(DateTime, nullable=False)
    subscription_end_date = Column(DateTime, nullable=True)
    repeat_pattern_id = Column(Integer, ForeignKey("repeat_patterns.id"), nullable=True)
    status = Column(String(50), nullable=True)  # active, cancelled, expired, suspended
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", back_populates="subscription_history")
    subscription_plan = relationship("SubscriptionPlan", back_populates="subscription_history")
    repeat_pattern = relationship("RepeatPattern", back_populates="subscription_history") 