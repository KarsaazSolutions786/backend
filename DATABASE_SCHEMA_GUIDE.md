# Database Schema & API Guide

## Database Configuration

**Database Name:** `eindr`  
**Connection:** PostgreSQL with SQLAlchemy ORM

## Database Schema

### 1. `users` Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    language VARCHAR,
    timezone VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. `preferences` Table
```sql
CREATE TABLE preferences (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    allow_friends BOOLEAN,
    receive_shared_notes BOOLEAN,
    notification_sound VARCHAR,
    tts_language VARCHAR,
    chat_history_enabled BOOLEAN
);
```

### 3. `reminders` Table
```sql
CREATE TABLE reminders (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) NOT NULL,
    title TEXT,
    description TEXT,
    time TIMESTAMP,
    repeat_pattern VARCHAR,
    timezone VARCHAR,
    is_shared BOOLEAN,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 4. `notes` Table
```sql
CREATE TABLE notes (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) NOT NULL,
    content TEXT,
    source VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 5. `ledger_entries` Table
```sql
CREATE TABLE ledger_entries (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) NOT NULL,
    contact_name VARCHAR,
    amount NUMERIC,
    direction VARCHAR CHECK (direction IN ('owe', 'owed')),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 6. `friendships` Table
```sql
CREATE TABLE friendships (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) NOT NULL,
    friend_id UUID REFERENCES users(id) NOT NULL,
    status VARCHAR CHECK (status IN ('pending', 'accepted', 'blocked')),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 7. `permissions` Table
```sql
CREATE TABLE permissions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) NOT NULL,
    friend_id UUID REFERENCES users(id) NOT NULL,
    auto_accept_reminders BOOLEAN,
    auto_accept_notes BOOLEAN,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 8. `embeddings` Table
```sql
CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) NOT NULL,
    reminder_id UUID REFERENCES reminders(id),
    embedding FLOAT8[],
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 9. `history_logs` Table
```sql
CREATE TABLE history_logs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) NOT NULL,
    content TEXT,
    interaction_type VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## API Endpoints

### Authentication (`/api/v1/auth`)
- `POST /register` - Register new user with Firebase auth
- `GET /me` - Get current user info with preferences
- `PUT /preferences` - Update user preferences

### Users (`/api/v1/users`)
- User management endpoints

### Reminders (`/api/v1/reminders`)
- `POST /` - Create reminder
- `GET /` - Get user reminders (filter by `shared`)
- `GET /{reminder_id}` - Get specific reminder
- `PUT /{reminder_id}` - Update reminder
- `DELETE /{reminder_id}` - Delete reminder
- `GET /upcoming/today` - Get today's reminders

### Notes (`/api/v1/notes`)
- `POST /` - Create note
- `GET /` - Get user notes (filter by `source`)
- `GET /{note_id}` - Get specific note
- `PUT /{note_id}` - Update note
- `DELETE /{note_id}` - Delete note
- `GET /search/{query}` - Search notes by content

### Ledger (`/api/v1/ledger`)
- `POST /` - Create ledger entry
- `GET /` - Get ledger entries (filter by `direction`, `contact_name`)
- `GET /{entry_id}` - Get specific entry
- `PUT /{entry_id}` - Update entry
- `DELETE /{entry_id}` - Delete entry
- `GET /summary/balance` - Get balance summary

### Friends (`/api/v1/friends`)
- `POST /` - Send friend request
- `GET /` - Get friendships (filter by `status_filter`)
- `PUT /{friendship_id}` - Update friendship status
- `DELETE /{friendship_id}` - Delete friendship
- `POST /permissions` - Create/update friend permissions
- `GET /permissions` - Get user permissions

### Embeddings (`/api/v1/embeddings`)
- `POST /` - Create embedding
- `GET /` - Get embeddings (filter by `reminder_id`)
- `GET /{embedding_id}` - Get specific embedding
- `DELETE /{embedding_id}` - Delete embedding
- `POST /search` - Search similar embeddings

### History (`/api/v1/history`)
- `POST /` - Create history log
- `GET /` - Get history logs (filter by `interaction_type`, `days`, `limit`)
- `GET /{log_id}` - Get specific log
- `DELETE /{log_id}` - Delete log
- `DELETE /` - Clear history logs (with filters)
- `GET /stats/summary` - Get history statistics

### Speech-to-Text (`/api/v1/stt`)
- STT processing endpoints

## Key Features

### Authentication
- Firebase Authentication integration
- No local password storage
- JWT token verification through Firebase

### Data Models
- All IDs are UUIDs
- Proper foreign key relationships
- Cascade delete for related records
- Timestamp tracking for creation/updates

### Validation
- Pydantic models for request/response validation
- Database constraints for data integrity
- Optional fields properly handled

### Error Handling
- Comprehensive HTTP status codes
- Detailed error messages
- Logging for debugging

## Example Usage

### Create a Reminder
```json
POST /api/v1/reminders
{
    "title": "Doctor Appointment",
    "description": "Annual checkup with Dr. Smith",
    "time": "2024-01-15T10:00:00",
    "repeat_pattern": "yearly",
    "timezone": "America/New_York",
    "is_shared": false
}
```

### Create a Ledger Entry
```json
POST /api/v1/ledger
{
    "contact_name": "John Doe",
    "amount": 25.50,
    "direction": "owe"
}
```

### Send Friend Request
```json
POST /api/v1/friends
{
    "friend_id": "user-uuid-here",
    "status": "pending"
}
```

## Database Connection

The project uses:
- **Database:** `eindr`
- **Connection String:** `postgresql://postgres:admin123@localhost:5432/eindr`
- **ORM:** SQLAlchemy with declarative base
- **Session Management:** FastAPI dependency injection

All models are properly defined with relationships and constraints matching your exact schema specifications. 