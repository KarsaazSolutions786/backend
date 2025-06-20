# Eindr Admin Panel - Complete Workflow Guide

## 🎯 Overview

The Eindr Admin Panel is a comprehensive, production-ready administrative interface built on FastAPI with enterprise-grade security, performance optimization, and full audit capabilities. This guide covers the complete workflow for administrators managing the Eindr AI-powered reminder application.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Authentication Workflow](#-authentication-workflow)
- [Role-Based Access Control](#-role-based-access-control)
- [Dashboard & Analytics](#-dashboard--analytics)
- [User Management](#-user-management)
- [API Reference](#-api-reference)
- [Security Features](#-security-features)
- [Performance & Caching](#-performance--caching)
- [Monitoring & Auditing](#-monitoring--auditing)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Redis (optional, for caching)
- Admin user account

### Initial Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export ADMIN_JWT_SECRET="your-secret-key"
export DATABASE_URL="postgresql://user:pass@host:port/db"
export REDIS_URL="redis://localhost:6379"

# 3. Run database migrations
alembic upgrade head

# 4. Create first admin user
python scripts/seed_admin.py

# 5. Start the server
python main.py
```

### First Login
```bash
# Test login with default credentials
curl -X POST http://localhost:8000/api/v1/admin/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "admin123"
  }'
```

## 🔐 Authentication Workflow

### 1. Admin Login Process

**Endpoint**: `POST /api/v1/admin/auth/login`

```json
{
  "email": "admin@example.com",
  "password": "your-password",
  "two_fa_token": "123456", // Optional if 2FA enabled
  "remember_me": false      // Extends token expiration
}
```

**Security Features**:
- ✅ Account lockout after 5 failed attempts (30-minute lockout)
- ✅ Password strength validation (12+ chars, complexity)
- ✅ Rate limiting (1000 requests/hour per admin)
- ✅ JWT tokens with role-based claims
- ✅ Session tracking with IP and user agent logging

**Response**:
```json
{
  "success": true,
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 28800,
  "admin_user": {
    "id": "uuid",
    "name": "Admin Name",
    "email": "admin@example.com",
    "role": "super_admin"
  },
  "session_info": {
    "login_time": "2024-01-01T12:00:00Z",
    "ip_address": "192.168.1.1"
  }
}
```

### 2. Two-Factor Authentication (2FA)

#### Setup 2FA
```bash
# Generate QR code and backup codes
curl -X POST http://localhost:8000/api/v1/admin/auth/setup-2fa \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Verify 2FA Setup
```bash
curl -X POST http://localhost:8000/api/v1/admin/auth/verify-2fa \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"token": "123456"}'
```

#### Disable 2FA
```bash
curl -X POST http://localhost:8000/api/v1/admin/auth/disable-2fa \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"password": "your-password"}'
```

### 3. Session Management

```bash
# Get admin profile
GET /api/v1/admin/auth/profile

# View active sessions
GET /api/v1/admin/auth/sessions

# Refresh access token
POST /api/v1/admin/auth/refresh-token

# Logout
POST /api/v1/admin/auth/logout
```

## 👥 Role-Based Access Control (RBAC)

### Permission Hierarchy

| Role | Level | Permissions |
|------|-------|-------------|
| **🔴 Super Admin** | 4 | Full system access, user deletion, admin management |
| **🟡 Support Agent** | 3 | User management (except delete), notifications |
| **🔵 Analyst** | 2 | Analytics, reporting, data export |
| **🟢 Read Only** | 1 | View-only dashboard access |

### Role-Based Endpoints

```python
# Super Admin only
@require_admin_roles([AdminRole.SUPER_ADMIN])
DELETE /api/v1/admin/users/{user_id}

# Support Agent or higher
@require_admin_roles([AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT])
POST /api/v1/admin/users/{user_id}/suspend

# Analyst or higher
@require_admin_roles([AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT, AdminRole.ANALYST])
GET /api/v1/admin/dashboard/growth-metrics

# Any admin role
@require_admin_roles([AdminRole.SUPER_ADMIN, AdminRole.SUPPORT_AGENT, AdminRole.ANALYST, AdminRole.READ_ONLY])
GET /api/v1/admin/dashboard/kpis
```

## 📊 Dashboard & Analytics

### 1. Main Dashboard KPIs

**Endpoint**: `GET /api/v1/admin/dashboard/kpis`

**Real-time Metrics**:
- 👥 User statistics (total, active, new)
- 📝 Content metrics (reminders, notes, ledger entries)
- 🤖 AI pipeline performance
- 📈 Growth indicators
- ⚡ System health status

**Example Response**:
```json
{
  "success": true,
  "data": {
    "overview": {
      "total_users": 1250,
      "active_users_today": 980,
      "new_users_today": 15,
      "trial_users": 320,
      "premium_users": 180,
      "ai_requests_today": 1200,
      "ai_success_rate": 95.0,
      "retention_rate_7d": 75.2
    },
    "performance": {
      "avg_response_time_ms": 245.0,
      "error_rate_percentage": 1.2,
      "status": "healthy"
    }
  }
}
```

### 2. Usage Analytics

**Endpoint**: `GET /api/v1/admin/dashboard/usage-analytics?hours=24`

**Features**:
- 📊 Hourly usage patterns (1-168 hours)
- 🔢 API request volumes
- 🤖 AI pipeline utilization
- 📱 Feature adoption tracking
- ⏱️ Response time analysis

### 3. System Health Monitoring

**Endpoint**: `GET /api/v1/admin/dashboard/system-health`

**Health Indicators**:
- 🟢 **Healthy**: Error rate < 1%, Response time < 500ms
- 🟡 **Warning**: Error rate 1-5%, Response time 500-1000ms
- 🔴 **Critical**: Error rate > 5%, Response time > 1000ms

### 4. Growth Analytics

**Endpoint**: `GET /api/v1/admin/dashboard/growth-metrics?period=weekly`

**Metrics**:
- 📈 User acquisition trends
- 🔄 Retention analysis (7d/30d)
- 💡 Engagement patterns
- 📊 Growth projections

## 👤 User Management

### 1. User Statistics Overview

```bash
GET /api/v1/admin/users/stats
```

**Response**:
```json
{
  "total_users": 1250,
  "active_users": 980,
  "new_users_today": 15,
  "trial_users": 320,
  "premium_users": 180,
  "suspended_users": 12,
  "avg_content_per_user": 8.5,
  "retention_rate_7d": 75.2
}
```

### 2. Advanced User Search

```bash
GET /api/v1/admin/users/search?page=1&limit=20&sort_by=created_at&sort_order=desc
```

**Search Filters**:
- 🔍 **Text Search**: Name, email, or ID
- 📧 **Email Domain**: Filter by domain
- 🏷️ **Account Status**: active, inactive, suspended, pending
- 💳 **Subscription**: trial, premium, free, expired
- 📅 **Date Ranges**: Created after/before, last active
- 📊 **Content Count**: Minimum content threshold
- ✅ **Trial Status**: Completed/not completed

**Example**:
```bash
curl "http://localhost:8000/api/v1/admin/users/search?account_status=active&subscription_status=trial&page=1&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. User Details & Management

#### Get User Details
```bash
GET /api/v1/admin/users/{user_id}?include_content=true&include_analytics=true
```

#### Update User Account
```bash
PUT /api/v1/admin/users/{user_id}
Content-Type: application/json

{
  "account_status": "active",
  "subscription_status": "premium",
  "notes": "Upgraded to premium manually",
  "force_password_reset": false
}
```

#### Suspend User
```bash
POST /api/v1/admin/users/{user_id}/suspend
Content-Type: application/json

{
  "reason": "Violation of terms of service",
  "duration_days": 30
}
```

#### Reactivate User
```bash
POST /api/v1/admin/users/{user_id}/reactivate
Content-Type: application/json

{
  "reason": "Issue resolved, account restored"
}
```

#### Delete User (Super Admin Only)
```bash
DELETE /api/v1/admin/users/{user_id}
Content-Type: application/json

{
  "reason": "GDPR deletion request",
  "confirm_deletion": true
}
```

### 4. Bulk Operations

**Endpoint**: `POST /api/v1/admin/users/bulk-actions`

**Supported Actions**:
- ✅ `activate` - Activate user accounts
- ❌ `deactivate` - Deactivate user accounts
- 🚫 `suspend` - Suspend user accounts
- 🗑️ `delete` - Delete accounts (Super Admin only)
- 📧 `send_notification` - Send bulk notifications

**Example**:
```json
{
  "user_ids": ["user1", "user2", "user3"],
  "action": "suspend",
  "reason": "Bulk suspension for policy violation",
  "notification_data": {
    "title": "Account Suspended",
    "message": "Your account has been temporarily suspended."
  }
}
```

### 5. Data Export

```bash
GET /api/v1/admin/users/export?format=csv&account_status=active
```

**Export Options**:
- 📄 **Formats**: CSV, JSON
- 🔍 **Filtering**: Apply same filters as search
- 📊 **Use Cases**: GDPR compliance, analytics, reporting

## 📚 API Reference

### Authentication Endpoints

| Method | Endpoint | Description | Required Role |
|--------|----------|-------------|---------------|
| POST | `/admin/auth/login` | Admin login | None |
| POST | `/admin/auth/logout` | Admin logout | Any |
| POST | `/admin/auth/setup-2fa` | Setup 2FA | Any |
| POST | `/admin/auth/verify-2fa` | Verify 2FA | Any |
| GET | `/admin/auth/profile` | Get profile | Any |

### Dashboard Endpoints

| Method | Endpoint | Description | Required Role |
|--------|----------|-------------|---------------|
| GET | `/admin/dashboard/kpis` | Dashboard KPIs | Any |
| GET | `/admin/dashboard/usage-analytics` | Usage analytics | Analyst+ |
| GET | `/admin/dashboard/system-health` | System health | Any |
| GET | `/admin/dashboard/growth-metrics` | Growth metrics | Analyst+ |
| POST | `/admin/dashboard/cache/refresh` | Refresh cache | Analyst+ |

### User Management Endpoints

| Method | Endpoint | Description | Required Role |
|--------|----------|-------------|---------------|
| GET | `/admin/users/stats` | User statistics | Analyst+ |
| GET | `/admin/users/search` | Search users | Support+ |
| GET | `/admin/users/{id}` | User details | Support+ |
| PUT | `/admin/users/{id}` | Update user | Support+ |
| POST | `/admin/users/{id}/suspend` | Suspend user | Support+ |
| DELETE | `/admin/users/{id}` | Delete user | Super Admin |
| POST | `/admin/users/bulk-actions` | Bulk operations | Support+ |
| GET | `/admin/users/export` | Export data | Analyst+ |

## 🛡️ Security Features

### Password Requirements
- ✅ Minimum 12 characters
- ✅ At least one uppercase letter
- ✅ At least one lowercase letter
- ✅ At least one number
- ✅ At least one special character

### Account Security
- 🔒 Account lockout after 5 failed attempts
- ⏰ 30-minute lockout duration
- 🔑 JWT tokens with role-based claims
- 📱 Two-factor authentication (TOTP)
- 🔄 Session management and tracking

### Request Security
- 🚦 Rate limiting (1000 requests/hour per admin)
- 🌐 IP address tracking
- 🖥️ User agent logging
- 🔐 Security headers (OWASP compliance)
- 📝 Comprehensive audit logging

## ⚡ Performance & Caching

### Caching Strategy

**Redis Primary + Memory Fallback**:
- 🎯 **KPI Data**: 60-second TTL
- 📊 **Usage Stats**: 5-minute TTL
- 📈 **Growth Metrics**: 1-hour TTL
- 🔧 **Performance Data**: 2-minute TTL

### Cache Management

```bash
# Get cache statistics
GET /api/v1/admin/dashboard/cache/stats

# Refresh cache manually
POST /api/v1/admin/dashboard/cache/refresh

# Refresh specific pattern
POST /api/v1/admin/dashboard/cache/refresh?pattern=dashboard
```

### Performance Monitoring

**Health Thresholds**:
- 🟢 **Excellent**: Cache hit rate > 90%
- 🟡 **Good**: Cache hit rate 70-90%
- 🔴 **Needs Improvement**: Cache hit rate < 70%

## 📊 Monitoring & Auditing

### Audit Logging

**Every admin action is logged with**:
- 👤 **Who**: Admin ID, name, role
- 📝 **What**: Action type (CREATE, UPDATE, DELETE, VIEW, EXPORT)
- ⏰ **When**: Timestamp with timezone
- 🌐 **Where**: IP address, user agent
- 🎯 **Target**: Resource type and ID
- ✅ **Result**: Success/failure with details
- 📄 **Context**: Request payload, description

### Monitoring Endpoints

```bash
# System health check
GET /api/v1/admin/dashboard/system-health

# Cache performance
GET /api/v1/admin/dashboard/cache/stats

# Usage patterns
GET /api/v1/admin/dashboard/usage-analytics?hours=24
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Check token validity
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/admin/auth/profile
```

**Solutions**:
- Verify token hasn't expired
- Check for account lockout
- Ensure 2FA token is correct if enabled

#### 2. Permission Denied
```json
{
  "detail": "Insufficient permissions. Required roles: ['super_admin']"
}
```

**Solutions**:
- Check admin role level
- Verify endpoint permission requirements
- Contact super admin for role upgrade

#### 3. Cache Issues
```bash
# Check cache health
GET /api/v1/admin/dashboard/cache/stats

# Refresh cache if needed
POST /api/v1/admin/dashboard/cache/refresh
```

#### 4. Performance Issues
```bash
# Monitor system health
GET /api/v1/admin/dashboard/system-health

# Check usage patterns
GET /api/v1/admin/dashboard/usage-analytics
```

### Environment Variables

```bash
# Required
ADMIN_JWT_SECRET="your-secret-key-min-32-chars"
DATABASE_URL="postgresql://user:pass@host:port/db"

# Optional
REDIS_URL="redis://localhost:6379"
ADMIN_JWT_EXPIRATION_HOURS="8"
ADMIN_LOCKOUT_ATTEMPTS="5"
ADMIN_LOCKOUT_DURATION="30"
ADMIN_CACHE_TTL_SECONDS="60"
```

### Database Migration

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Check current version
alembic current
```

### Logging Configuration

**Log Levels**:
- 🔍 **DEBUG**: Detailed debugging information
- ℹ️ **INFO**: General operational messages
- ⚠️ **WARNING**: Warning conditions
- ❌ **ERROR**: Error conditions
- 🚨 **CRITICAL**: Critical errors requiring immediate attention

## 📞 Support & Best Practices

### Best Practices

1. **🔐 Security**:
   - Use strong, unique passwords for admin accounts
   - Enable 2FA for all admin users
   - Regularly rotate JWT secrets
   - Monitor audit logs for suspicious activity

2. **👥 User Management**:
   - Always provide reasons for user actions
   - Use bulk operations for efficiency
   - Regularly export data for backup
   - Follow GDPR guidelines for user deletion

3. **📊 Monitoring**:
   - Check dashboard KPIs daily
   - Monitor system health regularly
   - Set up alerts for critical thresholds
   - Review cache performance weekly

4. **🔄 Maintenance**:
   - Refresh cache during low traffic
   - Clean up expired sessions
   - Monitor database performance
   - Keep audit logs for compliance

### Emergency Procedures

#### Account Lockout Recovery
```bash
# Reset lockout via database (Super Admin)
python scripts/reset_admin_lockout.py admin@example.com
```

#### Cache Emergency Flush
```bash
# Clear all caches
POST /api/v1/admin/dashboard/cache/refresh
```

#### System Status Check
```bash
# Quick health check
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/admin/dashboard/system-health
```

---

## 📝 License & Support

This admin panel is part of the Eindr AI-powered reminder application. For technical support, feature requests, or bug reports, please contact the development team.

**Version**: 1.0.0  
**Last Updated**: 2024-06-20  
**Compatibility**: Python 3.8+, FastAPI 0.100+, PostgreSQL 12+

---

*For more detailed API documentation, visit `/docs` endpoint when the server is running.* 