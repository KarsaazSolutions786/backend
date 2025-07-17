# API Documentation

## Table of Contents
- [Auth Service](#auth-service)
- [Customer Service](#customer-service)
- [Reminder Service](#reminder-service)
- [Note Service](#note-service)
- [Ledger Service](#ledger-service)
- [Chat Service](#chat-service)

---

## Auth Service

### POST `/auth/register`
**Description:** Register a new customer.

**Request Body:**
```json
{
  "email": "string (email)",
  "password": "string (min 8, max 128, strong)",
  "confirm_password": "string",
  "full_name": "string",
  "gender": "string",
  "is_new": true
}
```
**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 3600,
  "customer": {
    "id": 1,
    "email": "string",
    "is_verified": true,
    "is_active": true,
    "created_at": "datetime",
    "last_login": "datetime|null"
  }
}
```

### POST `/auth/login`
**Description:** Authenticate a customer and get JWT tokens.

**Request Body:**
```json
{
  "email": "string (email)",
  "password": "string",
  "remember_me": false
}
```
**Response:** _Same as `/auth/register`_

### POST `/auth/refresh`
**Description:** Refresh access token using a refresh token.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```
**Response:** _Same as `/auth/register`_

### GET `/auth/me`
**Description:** Get current customer info and sessions.

**Response:**
```json
{
  "id": 1,
  "email": "string",
  "is_verified": true,
  "is_active": true,
  "created_at": "datetime",
  "last_login": "datetime|null",
  "login_attempts": 0,
  "locked_until": "datetime|null",
  "subscription_plan_id": 1,
  "sessions": [ ... ],
  "recent_login_attempts": [ ... ]
}
```

---

## Customer Service

### GET `/customers/`
**Description:** List customers (paginated, filterable).

**Query Params:** `page`, `limit`, `search`, `is_active`

**Response:**
```json
{
  "customers": [ ... ],
  "total": 100,
  "page": 1,
  "limit": 20,
  "pages": 5
}
```

### GET `/customers/me`
**Description:** Get current customer profile.

**Response:**
```json
{
  "id": 1,
  "email": "string",
  "is_verified": true,
  "is_active": true,
  "created_at": "datetime",
  "last_login": "datetime|null"
}
```

### PUT `/customers/me`
**Description:** Update current customer profile.

**Request Body:**
```json
{
  "email": "string (email)",
  "is_active": true,
  "is_verified": true,
  "subscription_plan_id": 1
}
```
**Response:** _Same as `/customers/me`_

---

## Reminder Service

### POST `/reminders/`
**Description:** Create a new reminder.

**Request Body:**
```json
{
  "title": "string",
  "description": "string|null",
  "time": "datetime",
  "repeat_pattern": "none|daily|weekly|monthly|yearly",
  "timezone": "string",
  "priority": "low|medium|high|urgent",
  "category": "string|null",
  "tags": ["string"]
}
```
**Response:**
```json
{
  "id": "string",
  "customer_id": "string",
  "title": "string",
  "description": "string|null",
  "time": "datetime",
  "repeat_pattern": "string",
  "timezone": "string",
  "priority": "string",
  "category": "string|null",
  "tags": ["string"],
  "is_completed": false,
  "completed_at": "datetime|null",
  "is_active": true,
  "created_at": "datetime",
  "updated_at": "datetime",
  "next_occurrence": "datetime|null",
  "occurrence_count": "string",
  "max_occurrences": "string|null"
}
```

### GET `/reminders/`
**Description:** List reminders (filterable, paginated).

**Response:** `List<ReminderResponse>`

---

## Note Service

### POST `/notes/`
**Description:** Create a new note.

**Request Body:**
```json
{
  "title": "string",
  "description": "string",
  "content_type": "string",
  "is_favorite": false,
  "is_pinned": false
}
```
**Response:**
```json
{
  "id": 1,
  "customer_id": 1,
  "title": "string",
  "description": "string",
  "content_type": "string",
  "is_favorite": false,
  "is_pinned": false,
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### GET `/notes/`
**Description:** List notes (filterable, paginated).

**Response:**
```json
{
  "notes": [ ... ],
  "total": 100,
  "page": 1,
  "limit": 20,
  "has_next": true,
  "has_prev": false
}
```

---

## Ledger Service

### POST `/ledger-entries/`
**Description:** Create a new ledger entry.

**Request Body:**
```json
{
  "friend_id": 2,
  "amount": 100.0,
  "ledger_direction_id": 1,
  "notes": "string|null"
}
```
**Response:**
```json
{
  "id": 1,
  "customer_id": 1,
  "friend_id": 2,
  "amount": 100.0,
  "ledger_direction_id": 1,
  "notes": "string|null",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### GET `/ledger-entries/`
**Description:** List all ledger entries for current customer.

**Response:** `List<LedgerEntryResponse>`

---

## Chat Service

### POST `/chat`
**Description:** Main chat endpoint for AI conversations.

**Request Body:**
```json
{
  "message": "string",
  "conversation_id": "string|null",
  "customer_id": "string",
  "context": {"key": "value"},
  "temperature": 0.8,
  "max_length": 150
}
```
**Response:**
```json
{
  "response": "string",
  "conversation_id": "string",
  "message_id": "string",
  "timestamp": "datetime",
  "context": {"key": "value"},
  "model_info": {"key": "value"}
}
```

---

**For more endpoints and details, see each service's `/docs` endpoint.** 