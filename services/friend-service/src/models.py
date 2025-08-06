from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Self-referencing friendship relationships
    sent_friendships = relationship(
        "Friendship", 
        foreign_keys="Friendship.customer_id", 
        back_populates="customer",
        cascade="all, delete-orphan"
    )
    received_friendships = relationship(
        "Friendship", 
        foreign_keys="Friendship.friend_id", 
        back_populates="friend",
        cascade="all, delete-orphan"
    )
    
    # Friend permissions
    given_permissions = relationship(
        "FriendPermission", 
        foreign_keys="FriendPermission.customer_id", 
        back_populates="customer",
        cascade="all, delete-orphan"
    )
    received_permissions = relationship(
        "FriendPermission", 
        foreign_keys="FriendPermission.friend_id", 
        back_populates="friend",
        cascade="all, delete-orphan"
    )
    
    # Friend request history
    sent_requests = relationship(
        "FriendRequestHistory", 
        foreign_keys="FriendRequestHistory.requester_id", 
        back_populates="requester",
        cascade="all, delete-orphan"
    )
    received_requests = relationship(
        "FriendRequestHistory", 
        foreign_keys="FriendRequestHistory.requested_id", 
        back_populates="requested",
        cascade="all, delete-orphan"
    )

class Friendship(Base):
    __tablename__ = "friendships"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    status = Column(String, nullable=True, default="pending")  # 'pending', 'accepted', 'blocked', 'declined'
    initiated_by = Column(String, nullable=True)  # 'customer' or 'friend'
    message = Column(Text, nullable=True)  # Optional message when sending friend request
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    accepted_at = Column(DateTime, nullable=True)
    
    # Relationships
    customer = relationship("Customer", foreign_keys=[customer_id], back_populates="sent_friendships")
    friend = relationship("Customer", foreign_keys=[friend_id], back_populates="received_friendships")
    
    # Ensure unique friendship pairs
    __table_args__ = (
        {"sqlite_autoincrement": True}
    )

class FriendPermission(Base):
    __tablename__ = "friend_permissions"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    
    # Permission types
    can_see_notes = Column(Boolean, default=False)
    can_see_reminders = Column(Boolean, default=False)
    can_see_activity = Column(Boolean, default=False)
    can_see_location = Column(Boolean, default=False)
    can_share_notes = Column(Boolean, default=False)
    can_share_reminders = Column(Boolean, default=False)
    can_edit_shared_notes = Column(Boolean, default=False)
    can_edit_shared_reminders = Column(Boolean, default=False)
    can_see_profile = Column(Boolean, default=True)
    can_send_messages = Column(Boolean, default=True)
    
    # Permission metadata
    permission_level = Column(String, default="basic")  # 'basic', 'trusted', 'full', 'custom'
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    customer = relationship("Customer", foreign_keys=[customer_id], back_populates="given_permissions")
    friend = relationship("Customer", foreign_keys=[friend_id], back_populates="received_permissions")

class FriendRequestHistory(Base):
    __tablename__ = "friend_request_history"
    
    id = Column(Integer, primary_key=True, index=True)
    requester_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    requested_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    action = Column(String, nullable=False)  # 'sent', 'accepted', 'declined', 'blocked', 'canceled', 'unblocked'
    message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Additional metadata
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    # Relationships
    requester = relationship("Customer", foreign_keys=[requester_id], back_populates="sent_requests")
    requested = relationship("Customer", foreign_keys=[requested_id], back_populates="received_requests")