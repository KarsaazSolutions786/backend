# FastAPI Microservices Architecture Guide

## Overview

This document outlines the complete FastAPI microservices architecture with CRUD operations, relationships, authentication, and advanced features like sharing, pagination, and bulk operations.

## 🏗️ Service Architecture

### Core Services

1. **Auth Service** - Customer authentication, sessions, JWT tokens
2. **User Service** - Customer profiles, preferences, devices, subscriptions
3. **Note Service** - Notes management with sharing capabilities
4. **Reminder Service** - Reminders with scheduling and notifications
5. **Friend Service** - Friend relationships and permissions
6. **Ledger Service** - Expense tracking and splitting
7. **Chat Service** - AI conversations and message history

---

## 🔐 Authentication Service

**Base URL:** `/auth`

### Core Endpoints

```python
POST /auth/register          # Register new customer
POST /auth/login            # Customer login
POST /auth/logout           # Logout and invalidate sessions
GET  /auth/me               # Get current customer info
PUT  /auth/me               # Update customer info
POST /auth/change-password  # Change password
```

### Session Management

```python
GET    /auth/sessions           # Get active sessions
DELETE /auth/sessions/{id}      # Revoke specific session
GET    /auth/login-attempts     # Get login history
```

### Example Usage

```python
# Register a new customer
{
    "email": "user@example.com",
    "password": "SecurePass123!"
}

# Response
{
    "access_token": "eyJ0eXAiOiJKV1Q...",
    "refresh_token": "eyJ0eXAiOiJKV1Q...",
    "token_type": "bearer",
    "expires_in": 3600,
    "customer": {
        "id": 1,
        "email": "user@example.com",
        "is_verified": false,
        "is_active": true,
        "created_at": "2024-01-01T00:00:00Z"
    }
}
```

---

## 👤 User Service

**Base URL:** `/customers`

### Customer Profiles

```python
GET  /customers/profile         # Get customer profile
PUT  /customers/profile         # Update profile
POST /customers/profile/avatar  # Upload avatar
```

### Preferences

```python
GET /customers/preferences      # Get customer preferences
PUT /customers/preferences      # Update preferences
```

### Devices

```python
GET    /customers/devices       # Get registered devices
POST   /customers/devices       # Register new device
PUT    /customers/devices/{id}  # Update device
DELETE /customers/devices/{id}  # Remove device
```

### Example Profile Update

```python
{
    "full_name": "John Doe",
    "user_name": "john_doe",
    "bio": "Software developer and coffee enthusiast",
    "phone_number": "+1234567890",
    "date_of_birth": "1990-01-01",
    "country": "USA",
    "city": "San Francisco"
}
```

---

## 📝 Note Service

**Base URL:** `/notes`

### CRUD Operations

```python
POST   /notes              # Create note
GET    /notes              # List notes (with filtering)
GET    /notes/{id}         # Get specific note
PUT    /notes/{id}         # Update note
DELETE /notes/{id}         # Delete note
```

### Advanced Features

```python
# Filtering and pagination
GET /notes?page=1&limit=20&is_favorite=true&search=meeting&sort_by=updated_at&sort_order=desc

# Sharing
POST   /notes/{id}/share           # Share note with another user
GET    /notes/{id}/shares          # Get note shares
PUT    /notes/shares/{share_id}    # Update share permissions
DELETE /notes/shares/{share_id}    # Revoke share

# Bulk operations
POST   /notes/bulk-update          # Bulk update multiple notes
DELETE /notes/bulk-delete          # Bulk delete notes

# Statistics
GET    /notes/stats                # Get note statistics
```

### Example Note Creation

```python
{
    "title": "Meeting Notes",
    "description": "# Team Meeting\n\n- Discussed project roadmap\n- Assigned tasks",
    "content_type": "markdown",
    "is_favorite": true,
    "is_pinned": false
}
```

### Sharing Example

```python
{
    "shared_with_id": 2,
    "permission_level": "write",
    "can_edit": true,
    "can_delete": false,
    "can_reshare": false,
    "expires_at": "2024-12-31T23:59:59Z"
}
```

---

## ⏰ Reminder Service

**Base URL:** `/reminders`

### Core Operations

```python
POST   /reminders              # Create reminder
GET    /reminders              # List reminders
GET    /reminders/{id}         # Get specific reminder
PUT    /reminders/{id}         # Update reminder
DELETE /reminders/{id}         # Delete reminder
POST   /reminders/{id}/complete # Mark as complete
```

### Advanced Features

```python
# Sharing
POST   /reminders/{id}/share       # Share reminder
GET    /reminders/{id}/shares      # Get reminder shares
PUT    /reminders/shares/{id}      # Update share permissions
DELETE /reminders/shares/{id}      # Revoke share

# Notifications
GET    /reminders/{id}/notifications # Get notification history
POST   /reminders/{id}/notify       # Send immediate notification

# Scheduling
GET    /reminders/upcoming          # Get upcoming reminders
GET    /reminders/overdue           # Get overdue reminders
```

### Example Reminder

```python
{
    "title": "Team Meeting",
    "description": "Weekly team sync meeting",
    "time": "2024-01-15T14:00:00Z",
    "repeat_pattern_id": 2,  # Weekly
    "timezone_id": 1,        # UTC
    "priority_id": 3,        # High
    "max_occurrence": 10
}
```

---

## 👥 Friend Service

**Base URL:** `/friends`

### Friend Management

```python
POST   /friends/request         # Send friend request
GET    /friends/requests        # Get pending requests
PUT    /friends/requests/{id}   # Accept/decline request
DELETE /friends/requests/{id}   # Cancel request

GET    /friends                 # List friends
DELETE /friends/{id}            # Remove friend
POST   /friends/{id}/block      # Block user
DELETE /friends/{id}/block      # Unblock user
```

### Permissions

```python
GET /friends/{id}/permissions   # Get friend permissions
PUT /friends/{id}/permissions   # Update permissions
```

### Example Friend Request

```python
{
    "friend_id": 2,
    "message": "Hi! I'd like to connect with you."
}
```

### Permission Levels

```python
{
    "permission_level": "trusted",
    "can_see_notes": true,
    "can_see_reminders": true,
    "can_share_notes": true,
    "can_edit_shared_notes": false,
    "can_see_profile": true
}
```

---

## 💰 Ledger Service

**Base URL:** `/ledger`

### Expense Management

```python
POST   /ledger/entries         # Create expense entry
GET    /ledger/entries         # List entries
GET    /ledger/entries/{id}    # Get specific entry
PUT    /ledger/entries/{id}    # Update entry
DELETE /ledger/entries/{id}    # Delete entry
```

### Settlements

```python
POST /ledger/entries/{id}/approve  # Approve expense
POST /ledger/entries/{id}/settle   # Mark as settled
POST /ledger/entries/{id}/dispute  # Dispute expense
```

### Balances

```python
GET /ledger/balances           # Get all balances
GET /ledger/balances/{friend_id} # Get balance with specific friend
GET /ledger/summary            # Get overall summary
```

### Example Expense

```python
{
    "friend_id": 2,
    "amount": 45.50,
    "currency": "USD",
    "title": "Dinner at Italian Restaurant",
    "notes": "Split equally between John and Jane",
    "category": "food",
    "transaction_date": "2024-01-10T19:30:00Z",
    "split_type": "equal",
    "ledger_direction_id": 1  # outgoing
}
```

---

## 💬 Chat Service

**Base URL:** `/chat`

### Conversations

```python
POST   /chat/conversations     # Create conversation
GET    /chat/conversations     # List conversations
GET    /chat/conversations/{id} # Get conversation
PUT    /chat/conversations/{id} # Update conversation
DELETE /chat/conversations/{id} # Delete conversation
```

### Messages

```python
POST /chat/conversations/{id}/messages  # Send message
GET  /chat/conversations/{id}/messages  # Get messages
PUT  /chat/messages/{id}               # Edit message
DELETE /chat/messages/{id}             # Delete message
```

### Advanced Features

```python
# File attachments
POST /chat/messages/{id}/attachments   # Upload attachment
GET  /chat/messages/{id}/attachments   # Get attachments

# Conversation sharing
POST /chat/conversations/{id}/share    # Share conversation
GET  /chat/conversations/shared        # Get shared conversations

# Templates
GET  /chat/templates                   # Get conversation templates
POST /chat/templates                   # Create template
```

---

## 🔗 Relationship Patterns

### One-to-One Relationships

```python
# Customer -> CustomerProfile
customer = session.query(Customer).options(joinedload(Customer.profile)).first()
profile = customer.profile

# Customer -> CustomerPreferences  
preferences = customer.preferences
```

### One-to-Many Relationships

```python
# Customer -> Notes
notes = customer.notes

# Customer -> Reminders
reminders = customer.reminders.filter(Reminder.is_active == True).all()
```

### Many-to-Many Relationships

```python
# Friendship (self-referencing)
# Get all friends where status is 'accepted'
friends = session.query(Customer).join(
    Friendship, 
    or_(
        and_(Friendship.customer_id == customer.id, Friendship.friend_id == Customer.id),
        and_(Friendship.friend_id == customer.id, Friendship.customer_id == Customer.id)
    )
).filter(Friendship.status == 'accepted').all()

# Note sharing
shared_notes = session.query(Note).join(NoteShare).filter(
    NoteShare.shared_with_id == customer.id,
    NoteShare.is_active == True
).all()
```

---

## 🔍 Advanced Query Patterns

### Filtering with Relationships

```python
# Get notes shared with current user that are favorites
shared_favorite_notes = session.query(Note).join(
    NoteShare, Note.id == NoteShare.note_id
).filter(
    and_(
        NoteShare.shared_with_id == current_customer_id,
        NoteShare.is_active == True,
        Note.is_favorite == True,
        or_(
            NoteShare.expires_at.is_(None),
            NoteShare.expires_at > datetime.utcnow()
        )
    )
).all()
```

### Aggregations and Statistics

```python
# Get conversation message counts
conversation_stats = session.query(
    Conversation.id,
    Conversation.title,
    func.count(ChatMessage.id).label('message_count'),
    func.max(ChatMessage.created_at).label('last_message_at')
).outerjoin(ChatMessage).group_by(Conversation.id).all()

# Get friend balance summaries
balance_summary = session.query(
    Customer.email,
    func.sum(LedgerEntry.amount).label('total_owed'),
    func.count(LedgerEntry.id).label('transaction_count')
).join(LedgerEntry, Customer.id == LedgerEntry.friend_id).group_by(Customer.id).all()
```

---

## 🛡️ Authentication Flow

### JWT Token Usage

```python
# Every protected endpoint requires:
headers = {
    "Authorization": "Bearer eyJ0eXAiOiJKV1Q..."
}

# Token payload structure:
{
    "sub": "customer_id",
    "exp": "expiration_timestamp",
    "iat": "issued_at",
    "type": "access_token"
}
```

### Permission Checking

```python
# Check if user can edit a note
def can_edit_note(customer_id: int, note_id: int, db: Session) -> bool:
    note = db.query(Note).filter(Note.id == note_id).first()
    
    # Owner can always edit
    if note.customer_id == customer_id:
        return True
    
    # Check if shared with edit permission
    share = db.query(NoteShare).filter(
        and_(
            NoteShare.note_id == note_id,
            NoteShare.shared_with_id == customer_id,
            NoteShare.can_edit == True,
            NoteShare.is_active == True,
            or_(
                NoteShare.expires_at.is_(None),
                NoteShare.expires_at > datetime.utcnow()
            )
        )
    ).first()
    
    return share is not None
```

---

## 📊 Pagination and Filtering

### Standard Pagination Pattern

```python
# All list endpoints support:
{
    "page": 1,           # Page number (starts at 1)
    "limit": 20,         # Items per page (max 100)
    "sort_by": "created_at",
    "sort_order": "desc"
}

# Response format:
{
    "items": [...],
    "total": 150,
    "page": 1,
    "limit": 20,
    "has_next": true,
    "has_prev": false
}
```

### Advanced Filtering

```python
# Notes filtering
GET /notes?is_favorite=true&content_type=markdown&search=meeting&shared_with_me=false

# Reminders filtering
GET /reminders?status=active&priority=high&from_date=2024-01-01&to_date=2024-01-31

# Ledger filtering
GET /ledger/entries?friend_id=2&status=pending&category=food&min_amount=10.00
```

---

## 🚀 Performance Optimization

### Database Indexes

```sql
-- Essential indexes for performance
CREATE INDEX idx_notes_customer_created ON notes(customer_id, created_at);
CREATE INDEX idx_reminders_customer_time ON reminders(customer_id, time);
CREATE INDEX idx_friendships_status ON friendships(customer_id, friend_id, status);
CREATE INDEX idx_ledger_entries_customers ON ledger_entries(customer_id, friend_id);
CREATE INDEX idx_chat_messages_conversation ON chat_messages(conversions_id, created_at);
```

### Eager Loading

```python
# Load related data efficiently
notes_with_shares = session.query(Note).options(
    joinedload(Note.customer),
    joinedload(Note.shares).joinedload(NoteShare.shared_with)
).filter(Note.customer_id == customer_id).all()
```

---

## 🔧 Error Handling

### Standard HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (invalid/missing token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `409` - Conflict (duplicate data)
- `422` - Unprocessable Entity (validation errors)
- `500` - Internal Server Error

### Error Response Format

```python
{
    "detail": "Note not found",
    "error_code": "NOTE_NOT_FOUND"
}

# Validation errors
{
    "detail": "Validation failed",
    "errors": [
        {
            "field": "email",
            "message": "Invalid email format"
        }
    ]
}
```

---

## 🧪 Testing Examples

### Unit Test Example

```python
def test_create_note():
    note_data = {
        "title": "Test Note",
        "description": "Test content",
        "content_type": "text"
    }
    
    response = client.post("/notes/", json=note_data, headers=auth_headers)
    
    assert response.status_code == 201
    assert response.json()["title"] == "Test Note"
    assert response.json()["customer_id"] == current_customer_id
```

### Integration Test Example

```python
def test_note_sharing_workflow():
    # Create note
    note = create_test_note()
    
    # Share with another user
    share_data = {
        "shared_with_id": friend_id,
        "permission_level": "write",
        "can_edit": True
    }
    
    share_response = client.post(f"/notes/{note.id}/share", json=share_data)
    assert share_response.status_code == 201
    
    # Verify friend can access
    friend_response = client.get(f"/notes/{note.id}", headers=friend_auth_headers)
    assert friend_response.status_code == 200
```

---

## 📈 Monitoring and Metrics

### Key Metrics to Track

- API response times
- Database query performance
- Authentication success/failure rates
- Feature usage (notes created, reminders set, etc.)
- Error rates by endpoint
- User engagement metrics

### Health Check Endpoints

```python
GET /health              # Overall system health
GET /health/db          # Database connectivity
GET /health/redis       # Redis connectivity (if used)
GET /health/external    # External service dependencies
```

This comprehensive guide provides the foundation for a robust FastAPI microservices architecture with proper relationships, authentication, and advanced features for your personal assistant application. 