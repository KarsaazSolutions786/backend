# Friend Service API Documentation

## Overview
The Friend Service manages friend relationships, friend requests, and social interactions between users in the Eindr application.

## Base URL
```
/friends
```

## Authentication
All endpoints require JWT authentication via Bearer token in the Authorization header.

## Endpoints

### 1. Send Friend Request
**POST** `/requests`

Send a friend request to another user by email.

**Request Body:**
```json
{
  "friend_email": "friend@example.com",
  "message": "Optional message"
}
```

**Response:**
```json
{
  "id": "123",
  "customer_id": "456",
  "friend_id": "789",
  "friend_name": "friend",
  "friend_email": "friend@example.com",
  "status": "pending",
  "created_at": "2024-12-19T10:00:00Z",
  "accepted_at": null
}
```

### 2. Get Friends List
**GET** `/`

Retrieve friends list with optional filtering and pagination.

**Query Parameters:**
- `status` (optional): Filter by status (`pending`, `accepted`, `blocked`, `declined`)
- `limit` (optional): Maximum number of results (1-100, default: 50)
- `offset` (optional): Number of results to skip (default: 0)

**Response:**
```json
[
  {
    "id": "123",
    "customer_id": "456",
    "friend_id": "789",
    "friend_name": "friend",
    "friend_email": "friend@example.com",
    "status": "accepted",
    "created_at": "2024-12-19T10:00:00Z",
    "accepted_at": "2024-12-19T10:05:00Z"
  }
]
```

### 3. Accept Friend Request
**PUT** `/requests/{friendship_id}/accept`

Accept a pending friend request.

**Response:**
```json
{
  "message": "Friend request accepted successfully"
}
```

### 4. Decline Friend Request
**DELETE** `/requests/{friendship_id}`

Decline a pending friend request.

**Response:**
```json
{
  "message": "Friend request declined successfully"
}
```

### 5. Block Friend
**PUT** `/requests/{friendship_id}/block`

Block a friend or friend request.

**Request Body:**
```json
{
  "reason": "Optional reason for blocking"
}
```

**Response:**
```json
{
  "message": "Friend blocked successfully"
}
```

### 6. Unblock Friend
**PUT** `/requests/{friendship_id}/unblock`

Unblock a previously blocked friend.

**Response:**
```json
{
  "message": "Friend unblocked successfully"
}
```

### 7. Cancel Friend Request
**DELETE** `/requests/{friendship_id}/cancel`

Cancel a sent friend request (only the sender can cancel).

**Response:**
```json
{
  "message": "Friend request canceled successfully"
}
```

### 8. Get Blocked Friends
**GET** `/blocked`

Retrieve list of blocked friends.

**Response:**
```json
[
  {
    "id": "123",
    "customer_id": "456",
    "friend_id": "789",
    "friend_name": "blocked_user",
    "friend_email": "blocked@example.com",
    "status": "blocked",
    "created_at": "2024-12-19T10:00:00Z",
    "accepted_at": null
  }
]
```

### 9. Get Incoming Friend Requests
**GET** `/requests/incoming`

Retrieve friend requests received by the current user.

**Response:**
```json
[
  {
    "id": "123",
    "customer_id": "456",
    "friend_id": "789",
    "friend_name": "requester",
    "friend_email": "requester@example.com",
    "status": "pending",
    "created_at": "2024-12-19T10:00:00Z",
    "accepted_at": null
  }
]
```

### 10. Get Outgoing Friend Requests
**GET** `/requests/outgoing`

Retrieve friend requests sent by the current user.

**Response:**
```json
[
  {
    "id": "123",
    "customer_id": "456",
    "friend_id": "789",
    "friend_name": "recipient",
    "friend_email": "recipient@example.com",
    "status": "pending",
    "created_at": "2024-12-19T10:00:00Z",
    "accepted_at": null
  }
]
```

### 11. Search Friends
**POST** `/search`

Search for users by email to send friend requests.

**Request Body:**
```json
{
  "query": "search@example.com",
  "limit": 10
}
```

**Response:**
```json
[
  {
    "id": "123",
    "customer_id": "456",
    "friend_id": "789",
    "friend_name": "search",
    "friend_email": "search@example.com",
    "status": "none",
    "created_at": "2024-12-19T10:00:00Z",
    "accepted_at": null
  }
]
```

### 12. Get Mutual Friends
**GET** `/mutual/{friend_id}`

Get mutual friends between current user and specified friend.

**Response:**
```json
{
  "mutual_friends": [
    {
      "id": "123",
      "customer_id": "456",
      "friend_id": "789",
      "friend_name": "mutual_friend",
      "friend_email": "mutual@example.com",
      "status": "accepted",
      "created_at": "2024-12-19T10:00:00Z",
      "accepted_at": "2024-12-19T10:05:00Z"
    }
  ],
  "count": 1
}
```

### 13. Get Friendship Statistics
**GET** `/stats`

Retrieve friendship statistics for the current user.

**Response:**
```json
{
  "total_friends": 25,
  "pending_requests": 3,
  "sent_requests": 2,
  "received_requests": 3,
  "blocked_friends": 1,
  "shared_reminders": 0,
  "shared_notes": 0,
  "mutual_friends": 15
}
```

### 14. Get Friend Request History
**GET** `/history`

Retrieve complete friend request history for the current user.

**Response:**
```json
[
  {
    "id": 1,
    "requester_id": 456,
    "requested_id": 789,
    "action": "sent",
    "message": "Let's be friends!",
    "created_at": "2024-12-19T10:00:00Z",
    "requester_email": "user@example.com",
    "requested_email": "friend@example.com"
  }
]
```

## Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `500 Internal Server Error`: Server error

## Friend Request Actions

The following actions are tracked in friend request history:
- `sent`: Friend request sent
- `accepted`: Friend request accepted
- `declined`: Friend request declined
- `blocked`: Friend blocked
- `unblocked`: Friend unblocked
- `canceled`: Friend request canceled by sender

## Friend Status Types

- `pending`: Friend request is pending
- `accepted`: Friend request has been accepted
- `blocked`: Friend has been blocked
- `declined`: Friend request has been declined
- `none`: No friendship relationship exists

## Error Handling

All endpoints return consistent error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse. Standard limits apply:
- 100 requests per minute for most endpoints
- 10 requests per minute for search endpoints

## Security Features

- JWT token authentication
- Input validation and sanitization
- SQL injection prevention
- Request logging with IP and user agent tracking
- Secure error handling without information leakage