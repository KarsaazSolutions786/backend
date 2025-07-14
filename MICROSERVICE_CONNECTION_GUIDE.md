# Microservice Connection Guide

## Overview
This guide explains how to connect your Railway API gateway to actual microservices instead of using mock implementations.

## 🏗️ Architecture Options

### Option 1: HTTP Client Approach (Current Implementation)
- **Pros**: Simple, works with any microservice, easy to debug
- **Cons**: Network latency, potential failures
- **Best for**: Most use cases

### Option 2: Message Queue (RabbitMQ/Redis)
- **Pros**: Asynchronous, reliable, decoupled
- **Cons**: More complex, requires message queue setup
- **Best for**: High-volume, event-driven systems

### Option 3: gRPC
- **Pros**: Fast, type-safe, efficient
- **Cons**: More complex setup, less flexible
- **Best for**: Performance-critical services

## 🚀 Current Implementation: HTTP Client

### 1. Microservice Client Setup

The `src/services/microservice_client.py` handles all HTTP communication:

```python
# Example usage
result = await microservice_client.get_customers(token)
result = await microservice_client.create_reminder(data, token)
```

### 2. Environment Variables

Set these in your Railway environment:

```env
# Microservice URLs
AUTH_SERVICE_URL=https://your-auth-service.railway.app
CUSTOMER_SERVICE_URL=https://your-customer-service.railway.app
REMINDER_SERVICE_URL=https://your-reminder-service.railway.app
NOTE_SERVICE_URL=https://your-note-service.railway.app
LEDGER_SERVICE_URL=https://your-ledger-service.railway.app
FRIEND_SERVICE_URL=https://your-friend-service.railway.app
CHAT_SERVICE_URL=https://your-chat-service.railway.app
```

## 📋 Deployment Steps

### Step 1: Deploy Individual Microservices

1. **Auth Service**:
   ```bash
   # Create new Railway project for auth service
   cd services/auth-service
   railway init
   railway up
   ```

2. **Customer Service**:
   ```bash
   cd services/customer-service
   railway init
   railway up
   ```

3. **Repeat for other services**:
   - Reminder Service
   - Note Service
   - Ledger Service
   - Friend Service
   - Chat Service

### Step 2: Get Service URLs

After deployment, get the URLs from Railway dashboard:
- `https://your-auth-service.railway.app`
- `https://your-customer-service.railway.app`
- etc.

### Step 3: Configure API Gateway

Update your main Railway project environment variables:

```env
AUTH_SERVICE_URL=https://your-auth-service.railway.app
CUSTOMER_SERVICE_URL=https://your-customer-service.railway.app
REMINDER_SERVICE_URL=https://your-reminder-service.railway.app
NOTE_SERVICE_URL=https://your-note-service.railway.app
LEDGER_SERVICE_URL=https://your-ledger-service.railway.app
FRIEND_SERVICE_URL=https://your-friend-service.railway.app
CHAT_SERVICE_URL=https://your-chat-service.railway.app
```

### Step 4: Deploy API Gateway

```bash
# Deploy the main API gateway
railway up
```

## 🔧 Testing the Connection

### 1. Test Service Health

```bash
# Test each service individually
curl https://your-auth-service.railway.app/health
curl https://your-customer-service.railway.app/health
```

### 2. Test Through API Gateway

```bash
# Test through the main API gateway
curl https://your-api-gateway.railway.app/health
curl https://your-api-gateway.railway.app/api/status
```

### 3. Test Authentication Flow

```bash
# 1. Register user
curl -X POST https://your-api-gateway.railway.app/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# 2. Login
curl -X POST https://your-api-gateway.railway.app/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# 3. Use token to access protected routes
curl -X GET https://your-api-gateway.railway.app/customers \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🛠️ Advanced Configuration

### 1. Circuit Breaker Pattern

Add circuit breaker for better reliability:

```python
# In microservice_client.py
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def _make_request(self, method, service_url, endpoint, ...):
    # existing code
```

### 2. Retry Logic

Add retry mechanism:

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _make_request_with_retry(self, ...):
    # existing code
```

### 3. Service Discovery

For dynamic service URLs:

```python
# Use service discovery instead of hardcoded URLs
async def get_service_url(self, service_name: str) -> str:
    # Query service registry or use environment variables
    return os.getenv(f"{service_name.upper()}_SERVICE_URL")
```

## 🔍 Monitoring and Debugging

### 1. Logging

The client includes comprehensive logging:

```python
logger.info(f"Making request to {service_url}")
logger.error(f"Service timeout: {service_url}")
```

### 2. Health Checks

Add health check endpoints:

```python
@app.get("/health/services")
async def service_health():
    """Check health of all microservices"""
    services = {
        "auth": await check_service_health("AUTH_SERVICE_URL"),
        "customer": await check_service_health("CUSTOMER_SERVICE_URL"),
        # ... other services
    }
    return services
```

### 3. Metrics

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

request_counter = Counter('microservice_requests_total', 'Total requests', ['service', 'endpoint'])
request_duration = Histogram('microservice_request_duration_seconds', 'Request duration', ['service'])
```

## 🚨 Error Handling

### 1. Timeout Handling

```python
# Configurable timeouts
self.timeout = float(os.getenv("SERVICE_TIMEOUT", "30.0"))
```

### 2. Fallback Responses

```python
async def get_customers_with_fallback(self, token: str):
    try:
        return await self.get_customers(token)
    except Exception:
        # Return cached data or default response
        return {"customers": [], "cached": True}
```

### 3. Graceful Degradation

```python
# In main.py routes
try:
    result = await microservice_client.get_customers(token)
    return result
except HTTPException as e:
    if e.status_code == 503:
        # Service unavailable - return cached data
        return {"customers": [], "status": "degraded"}
    raise
```

## 📊 Performance Optimization

### 1. Connection Pooling

```python
# Use connection pooling for better performance
import httpx

async with httpx.AsyncClient(
    timeout=self.timeout,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
) as client:
    # Make requests
```

### 2. Caching

```python
# Add Redis caching
import redis.asyncio as redis

redis_client = redis.from_url(os.getenv("REDIS_URL"))

async def get_cached_data(self, key: str):
    cached = await redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None
```

## 🔐 Security Considerations

### 1. Service-to-Service Authentication

```python
# Add service authentication
headers = {
    "Authorization": f"Bearer {token}",
    "X-Service-Auth": os.getenv("SERVICE_AUTH_TOKEN")
}
```

### 2. Rate Limiting

```python
# Add rate limiting per service
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/customers")
@limiter.limit("100/minute")
async def create_customer(request: Request, ...):
    # existing code
```

## 🎯 Next Steps

1. **Deploy individual microservices** to Railway
2. **Configure environment variables** with service URLs
3. **Test the connections** using the provided examples
4. **Monitor performance** and add optimizations as needed
5. **Implement advanced patterns** (circuit breaker, caching, etc.)

Your API gateway is now ready to connect to actual microservices! 🚀 