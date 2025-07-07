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
    
    # Relationships
    notes = relationship("Note", back_populates="customer", cascade="all, delete-orphan")
    shared_notes = relationship("NoteShare", foreign_keys="NoteShare.shared_with_id", back_populates="shared_with")
    shared_by_notes = relationship("NoteShare", foreign_keys="NoteShare.shared_by_id", back_populates="shared_by")

class Note(Base):
    __tablename__ = "notes"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    content_type = Column(String, nullable=True, default="text")  # 'text', 'markdown', 'html'
    is_shared = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    last_accessed = Column(DateTime, nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="notes")
    shares = relationship("NoteShare", back_populates="note", cascade="all, delete-orphan")

class NoteShare(Base):
    __tablename__ = "note_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(Integer, ForeignKey("notes.id"), nullable=False)
    # Database uses owner_customer_id instead of shared_by_id
    shared_by_id = Column("owner_customer_id", Integer, ForeignKey("customers.id"), nullable=False)
    # Database uses shared_with_customer_id instead of shared_with_id
    shared_with_id = Column("shared_with_customer_id", Integer, ForeignKey("customers.id"), nullable=False)
    # Only columns present in the actual table
    can_edit = Column(Boolean, default=False)
    can_comment = Column(Boolean, default=False)
    responded_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    shared_at = Column(DateTime, default=func.current_timestamp())
    # Remove non-existent columns (permission_level, can_delete, can_reshare, expires_at, is_active)
    
    # Relationships
    note = relationship("Note", back_populates="shares")
    shared_by = relationship("Customer", foreign_keys=[shared_by_id], back_populates="shared_by_notes")
    shared_with = relationship("Customer", foreign_keys=[shared_with_id], back_populates="shared_notes") 