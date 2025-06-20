# Eindr Admin Panel Guide
*Production-Ready Admin Panel for Eindr FastAPI Backend*

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup & Installation](#setup--installation)
4. [Admin Roles & Permissions](#admin-roles--permissions)
5. [API Endpoints](#api-endpoints)
6. [Security Features](#security-features)
7. [KPI Dashboard](#kpi-dashboard)
8. [User Management](#user-management)
9. [Testing](#testing)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

## Overview

The Eindr Admin Panel is a comprehensive backend administration system built directly into the FastAPI monorepo. It provides secure, role-based access to manage users, monitor system performance, and maintain the application without requiring a separate admin service.

### Key Features
- ✅ **Role-Based Access Control (RBAC)** - 4 distinct admin roles
- ✅ **Real-Time Dashboard** - Live KPIs and performance metrics
- ✅ **User Management** - Advanced search, filtering, and bulk operations
- ✅ **Security** - 2FA, JWT tokens, audit logging, session management
- ✅ **Caching** - Redis-based KPI caching with fallback
- ✅ **Audit Trail** - Comprehensive logging of all admin actions
- ✅ **Data Export** - CSV/JSON export capabilities
- ✅ **Regression Protection** - Maintains existing AI pipeline functionality

## Architecture

### Database Schema
```sql
-- Admin Users Table
admin_users (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role AdminRole NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    two_fa_enabled BOOLEAN DEFAULT FALSE,
    two_fa_secret VARCHAR(32),
    backup_codes JSON,
    is_active BOOLEAN DEFAULT TRUE,
    last_login_at TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    password_changed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES admin_users(id)
)

-- Audit Log Table  
admin_audit_log (
    id UUID PRIMARY KEY,
    admin_id UUID REFERENCES admin_users(id),
    action AuditAction NOT NULL,
    target_type VARCHAR(50),
    target_id VARCHAR(255),
    payload JSON,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)

-- Additional tables: notifications, feature_flags, kpi_snapshots
```

### Role Hierarchy
1. **Super Admin** - Full system access, user management, feature flags
2. **Support Agent** - User CRUD, limited administrative functions
3. **Analyst** - Read-only access to dashboards and analytics
4. **Read Only** - View-only access to basic KPIs

## Setup & Installation

### 1. Install Dependencies
```bash
# Install new admin panel dependencies
pip install pyotp qrcode pillow redis

# Or update from requirements.txt
pip install -r requirements.txt
```

### 2. Database Migration
```bash
# Run Alembic migration to create admin tables
alembic upgrade head
```

### 3. Create First Super Admin
```bash
# Set environment variables
export FIRST_SUPERADMIN_EMAIL="admin@yourcompany.com"
export FIRST_SUPERADMIN_PASSWORD="YourSecurePassword123!"
export FIRST_SUPERADMIN_NAME="System Administrator"

# Run seed script
python scripts/seed_admin.py init
```

### 4. Optional: Create Sample Users
```bash
# Create sample admin users for testing
python scripts/seed_admin.py samples
```

### 5. Verify Installation
```bash
# Check admin panel status
python scripts/seed_admin.py status

# Start the server
python main.py
```

## Admin Roles & Permissions

### Super Admin (`super_admin`)
- ✅ Full system access
- ✅ Manage all admin users
- ✅ User account deletion
- ✅ Feature flag management  
- ✅ System configuration
- ✅ All dashboard and analytics access

### Support Agent (`support_agent`)
- ✅ User CRUD operations
- ✅ User suspension/reactivation
- ✅ Bulk user actions
- ✅ Dashboard access
- ❌ Cannot delete users permanently
- ❌ Cannot manage feature flags

### Analyst (`analyst`)
- ✅ Read-only dashboard access
- ✅ Data export capabilities
- ✅ Advanced analytics
- ❌ Cannot modify user accounts
- ❌ No administrative functions

### Read Only (`read_only`)
- ✅ Basic KPI viewing
- ❌ No user management
- ❌ No data modification
- ❌ Limited dashboard access

## API Endpoints

### Authentication Endpoints
```http
POST /api/v1/admin/auth/login           # Admin login with 2FA support
POST /api/v1/admin/auth/logout          # Admin logout
POST /api/v1/admin/auth/change-password # Change password
POST /api/v1/admin/auth/setup-2fa       # Setup two-factor authentication
POST /api/v1/admin/auth/verify-2fa      # Verify 2FA setup
POST /api/v1/admin/auth/disable-2fa     # Disable 2FA
GET  /api/v1/admin/auth/profile         # Get admin profile
GET  /api/v1/admin/auth/sessions        # Get active sessions
POST /api/v1/admin/auth/refresh-token   # Refresh access token
```

### Dashboard Endpoints
```http
GET  /api/v1/admin/dashboard/kpis           # Get dashboard KPIs
GET  /api/v1/admin/dashboard/usage-analytics # Get usage analytics
GET  /api/v1/admin/dashboard/system-health   # Get system health
GET  /api/v1/admin/dashboard/growth-metrics  # Get growth metrics
POST /api/v1/admin/dashboard/cache/refresh   # Refresh cache
GET  /api/v1/admin/dashboard/cache/stats     # Get cache statistics
GET  /api/v1/admin/dashboard/export          # Export dashboard data
```

### User Management Endpoints
```http
GET    /api/v1/admin/users/stats              # Get user statistics
GET    /api/v1/admin/users/search             # Search users with filters
GET    /api/v1/admin/users/{user_id}          # Get user details
PUT    /api/v1/admin/users/{user_id}          # Update user
POST   /api/v1/admin/users/{user_id}/suspend  # Suspend user
POST   /api/v1/admin/users/{user_id}/reactivate # Reactivate user
DELETE /api/v1/admin/users/{user_id}          # Delete user (Super Admin only)
POST   /api/v1/admin/users/bulk-actions       # Bulk user operations
GET    /api/v1/admin/users/export             # Export user data
```

## Security Features

### JWT Authentication
- **Token Expiration**: 8 hours (24 hours with "remember me")
- **Algorithm**: HS256 with secure secret key
- **Claims**: Admin ID, role, permissions, expiration

### Two-Factor Authentication (2FA)
- **TOTP**: Time-based One-Time Passwords
- **QR Codes**: Easy mobile app setup
- **Backup Codes**: Recovery options
- **Optional**: Can be enabled per admin user

### Account Security
- **Password Requirements**: Minimum 12 characters, complexity rules
- **Account Lockout**: 5 failed attempts = 15-minute lockout
- **Session Management**: Track and manage active sessions
- **Audit Logging**: Every action logged with IP and user agent

### Rate Limiting & Protection
- **Login Attempts**: Account lockout after failed attempts
- **IP Tracking**: Monitor suspicious activity
- **Session Validation**: Automatic token verification
- **CORS Configuration**: Proper cross-origin settings

## KPI Dashboard

### Performance Metrics
```json
{
  "overview": {
    "total_users": 1250,
    "active_users_today": 456,
    "new_users_today": 15,
    "retention_rate_7d": 75.2,
    "ai_requests_today": 234
  },
  "performance": {
    "avg_response_time_ms": 245,
    "error_rate_percentage": 0.8,
    "uptime_percentage": 99.9
  },
  "growth": {
    "current_week_new_users": 105,
    "previous_week_new_users": 89,
    "growth_rate_percentage": 18.0
  }
}
```

### Caching Strategy
- **Redis Primary**: High-performance caching with 60-second TTL
- **Memory Fallback**: In-memory cache when Redis unavailable
- **Cache Warming**: Pre-populate critical metrics
- **Invalidation**: Manual and automatic cache refresh

### Real-Time Updates
- **Live Metrics**: Updated every 60 seconds
- **Health Monitoring**: System status indicators
- **Performance Alerts**: Automatic threshold notifications

## User Management

### Advanced Search & Filtering
```javascript
// Search Parameters
{
  "search_query": "john@example.com",
  "account_status": "active",
  "subscription_status": "trial", 
  "created_after": "2024-01-01T00:00:00Z",
  "created_before": "2024-12-31T23:59:59Z",
  "min_content_count": 5
}

// Pagination & Sorting
{
  "page": 1,
  "limit": 20,
  "sort_by": "created_at",
  "sort_order": "desc"
}
```

### Bulk Operations
- **Activate/Deactivate**: Multiple users at once
- **Suspend**: Temporary account suspension
- **Send Notifications**: Broadcast messages
- **Data Export**: Filtered CSV/JSON exports
- **Audit Trail**: All bulk actions logged

### User Analytics
- **Content Statistics**: Reminders, notes, ledger entries
- **Usage Patterns**: Session duration, feature usage
- **Engagement Metrics**: Retention, activity levels
- **Device Information**: Platform, app version tracking

## Testing

### Regression Testing
The admin panel includes comprehensive regression tests to ensure the AI pipeline functionality remains intact:

```bash
# Run admin panel tests
pytest tests/admin/ -v

# Run regression tests for AI pipeline
pytest tests/admin/test_empty_audio_guard.py -v

# Run all tests
pytest -v
```

### Test Coverage
- **RBAC Security**: Role permission enforcement
- **Authentication**: JWT, 2FA, session management
- **User Management**: CRUD operations, bulk actions
- **Dashboard**: KPI calculations, caching
- **Audit Logging**: Action tracking, data integrity
- **Regression**: AI pipeline compatibility

### Load Testing
```bash
# Test admin endpoints under load
ab -n 1000 -c 10 http://localhost:8000/api/v1/admin/dashboard/kpis

# Monitor performance metrics
pytest tests/admin/test_performance.py
```

## Deployment

### Environment Variables
```bash
# Admin Panel Configuration
FIRST_SUPERADMIN_EMAIL=admin@yourcompany.com
FIRST_SUPERADMIN_PASSWORD=YourSecurePassword123!
FIRST_SUPERADMIN_NAME=System Administrator

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# Security Configuration
ADMIN_JWT_SECRET=your-secure-jwt-secret-key
ADMIN_SESSION_TIMEOUT=28800  # 8 hours in seconds
```

### Docker Deployment
```dockerfile
# Admin panel dependencies already included in main Dockerfile
# No additional Docker configuration required

# Environment variables in docker-compose.yml
environment:
  - FIRST_SUPERADMIN_EMAIL=admin@yourcompany.com
  - FIRST_SUPERADMIN_PASSWORD=SecurePassword123!
  - REDIS_URL=redis://redis:6379/0
```

### Production Checklist
- [ ] Admin JWT secret configured
- [ ] First super admin created
- [ ] Redis configured (optional but recommended)
- [ ] SSL/TLS enabled for admin endpoints
- [ ] Rate limiting configured
- [ ] Monitoring and alerting setup
- [ ] Backup procedures for admin data

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
```bash
# Install missing dependencies
pip install pyotp qrcode pillow redis

# Verify installations
python -c "import pyotp, qrcode, redis; print('All dependencies installed')"
```

#### 2. Database Migration Issues
```bash
# Check migration status
alembic current

# Force migration
alembic upgrade head

# If migration fails, check for existing tables
python -c "from connect_db import get_database; print('Database connected')"
```

#### 3. Super Admin Creation Fails
```bash
# Check environment variables
echo $FIRST_SUPERADMIN_EMAIL
echo $FIRST_SUPERADMIN_PASSWORD

# Verify password requirements
python scripts/seed_admin.py init
```

#### 4. 2FA Setup Issues
```bash
# Test TOTP generation
python -c "import pyotp; print(pyotp.TOTP('BASE32SECRET').now())"

# Check QR code generation
python -c "import qrcode; print('QR code library working')"
```

#### 5. Cache Performance Issues
```bash
# Check Redis connection
redis-cli ping

# Monitor cache statistics
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  http://localhost:8000/api/v1/admin/dashboard/cache/stats
```

### Debugging Tips

#### Enable Debug Logging
```python
import logging
logging.getLogger("core.admin_security").setLevel(logging.DEBUG)
logging.getLogger("routers.admin").setLevel(logging.DEBUG)
```

#### Check Audit Logs
```sql
-- View recent admin actions
SELECT * FROM admin_audit_log 
ORDER BY created_at DESC 
LIMIT 50;

-- Check for failed authentication attempts
SELECT admin_id, action, success, error_message, created_at
FROM admin_audit_log 
WHERE action = 'LOGIN' AND success = false
ORDER BY created_at DESC;
```

#### Monitor Performance
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s \
  -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/admin/dashboard/kpis
```

## Security Considerations

### Production Security Checklist
- [ ] Strong JWT secret (minimum 32 characters)
- [ ] HTTPS enabled for all admin endpoints
- [ ] Admin access restricted by IP (if possible)
- [ ] Regular security audits of admin actions
- [ ] 2FA enforced for super admins
- [ ] Regular password rotation policy
- [ ] Session timeout configured appropriately
- [ ] Rate limiting on authentication endpoints

### Compliance Features
- **GDPR Compliance**: User data export and deletion capabilities
- **Audit Requirements**: Comprehensive action logging
- **Data Privacy**: Admin access controls and permissions
- **Security Standards**: Industry-standard authentication and authorization

## API Documentation

The admin panel is fully documented in the FastAPI interactive docs:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

All admin endpoints are tagged as "Admin" and include detailed request/response schemas, authentication requirements, and permission levels.

---

**Admin Panel Version**: 1.0.0  
**FastAPI Integration**: Seamless monorepo integration  
**Production Ready**: ✅ Comprehensive testing and security  
**Maintenance**: Part of main Eindr application lifecycle 