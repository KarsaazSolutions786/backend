from pydantic import BaseModel, validator, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

# Enums
class ContentType(str, Enum):
    text = "text"
    markdown = "markdown"
    html = "html"

class PermissionLevel(str, Enum):
    read = "read"
    write = "write"
    admin = "admin"

# Request Schemas
class NoteCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    content_type: ContentType = ContentType.text
    is_favorite: Optional[bool] = False
    is_pinned: Optional[bool] = False

class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    content_type: Optional[ContentType] = None
    is_favorite: Optional[bool] = None
    is_pinned: Optional[bool] = None

class NoteShareCreate(BaseModel):
    shared_with_id: int
    permission_level: PermissionLevel = PermissionLevel.read
    can_edit: Optional[bool] = False
    can_delete: Optional[bool] = False
    can_reshare: Optional[bool] = False
    expires_at: Optional[datetime] = None

class NoteShareUpdate(BaseModel):
    permission_level: Optional[PermissionLevel] = None
    can_edit: Optional[bool] = None
    can_delete: Optional[bool] = None
    can_reshare: Optional[bool] = None
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None

# Response Schemas
class CustomerBase(BaseModel):
    id: int
    email: str
    
    class Config:
        from_attributes = True

class NoteShareResponse(BaseModel):
    id: int
    note_id: int
    shared_by_id: int
    shared_with_id: int
    permission_level: str
    can_edit: bool
    can_delete: bool
    can_reshare: bool
    shared_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    shared_by: CustomerBase
    shared_with: CustomerBase
    
    class Config:
        from_attributes = True

class NoteResponse(BaseModel):
    id: int
    customer_id: int
    title: Optional[str]
    description: Optional[str]
    content_type: str
    is_shared: bool
    is_favorite: bool
    is_pinned: bool
    created_at: datetime
    updated_at: datetime
    last_accessed: Optional[datetime]
    customer: CustomerBase
    
    class Config:
        from_attributes = True

class NoteWithShares(NoteResponse):
    shares: List[NoteShareResponse] = []

class NotesListResponse(BaseModel):
    notes: List[NoteResponse]
    total: int
    page: int
    limit: int
    has_next: bool
    has_prev: bool

# Filter Schemas
class NoteFilters(BaseModel):
    is_favorite: Optional[bool] = None
    is_pinned: Optional[bool] = None
    is_shared: Optional[bool] = None
    content_type: Optional[ContentType] = None
    search: Optional[str] = None
    shared_with_me: Optional[bool] = None

# Bulk Operations
class NoteBulkUpdate(BaseModel):
    note_ids: List[int]
    is_favorite: Optional[bool] = None
    is_pinned: Optional[bool] = None

class NoteBulkDelete(BaseModel):
    note_ids: List[int]

# Statistics
class NoteStats(BaseModel):
    total_notes: int
    favorite_notes: int
    pinned_notes: int
    shared_notes: int
    notes_shared_with_me: int
    notes_by_content_type: dict

# Error Schemas
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None 