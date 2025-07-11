from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import sqlalchemy.dialects.postgresql as pg

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
    api_usage_logs = relationship("ApiUsageLog", back_populates="customer", cascade="all, delete-orphan")

class ApiUsageLog(Base):
    __tablename__ = "api_usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    end_point = Column(String(255), nullable=True)  # API endpoint called
    method = Column(String(10), nullable=True)  # GET, POST, PUT, DELETE, etc.
    status_code = Column(Integer, nullable=True)  # HTTP status code
    response_time_ms = Column(Integer, nullable=True)  # Response time in milliseconds
    request_size_bytes = Column(Integer, nullable=True)  # Size of request in bytes
    ip_address = Column(pg.INET, nullable=True)  # Client IP address
    user_agent = Column(String(255), nullable=True)  # User agent string
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", back_populates="api_usage_logs") 