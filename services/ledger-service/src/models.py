from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func, ForeignKey, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships for ledger entries
    ledger_entries_as_customer = relationship(
        "LedgerEntry", 
        foreign_keys="LedgerEntry.customer_id", 
        back_populates="customer",
        cascade="all, delete-orphan"
    )
    ledger_entries_as_friend = relationship(
        "LedgerEntry", 
        foreign_keys="LedgerEntry.friend_id", 
        back_populates="friend",
        cascade="all, delete-orphan"
    )

class Friendship(Base):
    __tablename__ = "friendships"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    status = Column(String, nullable=False, default="pending")  # 'pending', 'accepted', 'blocked'
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    user = relationship("Customer", foreign_keys=[user_id])
    friend = relationship("Customer", foreign_keys=[friend_id])

class LedgerDirection(Base):
    __tablename__ = "ledger_direction"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)  # 'incoming', 'outgoing', 'settled'
    description = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    ledger_entries = relationship("LedgerEntry", back_populates="direction")

class LedgerEntry(Base):
    __tablename__ = "ledger_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    
    # For app users (friends)
    friend_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    
    # For non-app contacts
    friend_name = Column(String(255), nullable=True)
    friend_phone = Column(String(20), nullable=True)
    friend_email = Column(String(255), nullable=True)
    
    # Transaction details
    amount = Column(Numeric(10, 2), nullable=False)
    ledger_direction_id = Column(Integer, ForeignKey("ledger_direction.id"), nullable=False)
    notes = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="saved")  # 'draft', 'saved'
    
    # Metadata
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", foreign_keys=[customer_id], back_populates="ledger_entries_as_customer")
    friend = relationship("Customer", foreign_keys=[friend_id], back_populates="ledger_entries_as_friend")
    direction = relationship("LedgerDirection", back_populates="ledger_entries")

class LedgerEntryRelation(Base):
    __tablename__ = "ledger_entry_relations"
    id = Column(Integer, primary_key=True, index=True)
    # Disabled in simplified schema

class LedgerSummary(Base):
    """Cached summary of ledger balances between customers"""
    __tablename__ = "ledger_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    
    # Balance information
    total_owed_to_friend = Column(Numeric(10, 2), default=0)  # How much customer owes to friend
    total_owed_by_friend = Column(Numeric(10, 2), default=0)  # How much friend owes to customer
    net_balance = Column(Numeric(10, 2), default=0)  # Positive means friend owes customer, negative means customer owes friend
    currency = Column(String, default="USD")
    
    # Summary metadata
    last_transaction_date = Column(DateTime, nullable=True)
    total_transactions = Column(Integer, default=0)
    pending_transactions = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", foreign_keys=[customer_id])
    friend = relationship("Customer", foreign_keys=[friend_id])