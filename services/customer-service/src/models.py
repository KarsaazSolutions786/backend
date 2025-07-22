from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func, ForeignKey, Date, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    last_login = Column(DateTime, nullable=True)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    subscription_plan_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=True)
    
    # Relationships
    profile = relationship("CustomerProfile", back_populates="customer", uselist=False, cascade="all, delete-orphan")
    preferences = relationship("CustomerPreference", back_populates="customer", uselist=False, cascade="all, delete-orphan")
    subscription_plan = relationship("SubscriptionPlan", back_populates="customers")
    devices = relationship("CustomerDevice", back_populates="customer", cascade="all, delete-orphan")

class CustomerProfile(Base):
    __tablename__ = "customers_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False, unique=True)
    full_name = Column(String, nullable=True)
    user_name = Column(String, unique=True, nullable=True, index=True)
    bio = Column(Text, nullable=True)
    avatar_url = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    date_of_birth = Column(Date, nullable=True)
    timezone_id = Column(Integer, ForeignKey("timezones.id"), nullable=True)
    language_id = Column(Integer, ForeignKey("languages.id"), nullable=True)
    subscription_plan_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=True)
    country = Column(String, nullable=True)
    city = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    is_new = Column(Boolean, default=True)  # Add is_new column
    
    # Relationships
    customer = relationship("Customer", back_populates="profile")
    timezone = relationship("Timezone", back_populates="customer_profiles")
    language = relationship("Language", back_populates="customer_profiles")
    subscription_plan = relationship("SubscriptionPlan", back_populates="customer_profiles")

class CustomerPreference(Base):
    __tablename__ = "customer_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False, unique=True)
    allow_friends = Column(Boolean, default=True)
    received_shared_notes = Column(Boolean, default=True)
    notification_sound = Column(String, nullable=True)
    language_id = Column(Integer, ForeignKey("languages.id"), nullable=True)
    chat_history_enabled = Column(Boolean, default=True)
    theme = Column(String, nullable=True, default="light")
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=True)
    notification_frequency = Column(String, nullable=True, default="immediate")
    auto_backup = Column(Boolean, default=True)
    data_retention_days = Column(Integer, default=30)
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", back_populates="preferences")
    language = relationship("Language", back_populates="customer_preferences")

class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    price = Column(Numeric(10, 2), nullable=False)
    currency = Column(String, default="USD")
    duration_days = Column(Integer, nullable=False)
    max_reminders = Column(Integer, nullable=True)
    max_notes = Column(Integer, nullable=True)
    max_friends = Column(Integer, nullable=True)
    features = Column(Text, nullable=True)  # JSON string of features
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customers = relationship("Customer", back_populates="subscription_plan")
    customer_profiles = relationship("CustomerProfile", back_populates="subscription_plan")

class Timezone(Base):
    __tablename__ = "timezones"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    abbreviation = Column(String, nullable=True)
    utc_offset = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    customer_profiles = relationship("CustomerProfile", back_populates="timezone")

class Language(Base):
    __tablename__ = "languages"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, nullable=False, unique=True)  # e.g., 'en', 'es', 'fr'
    name = Column(String, nullable=False)  # e.g., 'English', 'Spanish', 'French'
    native_name = Column(String, nullable=True)  # e.g., 'English', 'Español', 'Français'
    is_active = Column(Boolean, default=True)
    
    # Relationships
    customer_profiles = relationship("CustomerProfile", back_populates="language")
    customer_preferences = relationship("CustomerPreference", back_populates="language")

class CustomerDevice(Base):
    __tablename__ = "customers_devices"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    device_type = Column(String, nullable=True)  # 'mobile', 'desktop', 'tablet'
    device_name = Column(String, nullable=True)
    device_id = Column(String, nullable=False, unique=True)
    push_token = Column(String, nullable=True)  # FCM token for push notifications
    last_active = Column(DateTime, default=func.current_timestamp())
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", back_populates="devices") 