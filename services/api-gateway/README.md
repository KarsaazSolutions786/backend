# Kong API Gateway for Eindr Microservices

This directory contains the Kong API Gateway configuration for the Eindr microservices architecture.

## Overview

Kong serves as the central API Gateway, providing:

- **Unified Entry Point**: Single endpoint for all microservices
- **Authentication**: JWT validation for protected routes
- **Rate Limiting**: Per-service traffic control
- **CORS**: Cross-origin resource sharing
- **Logging**: Request/response logging and monitoring
- **Load Balancing**: Automatic failover and distribution

## Architecture

```
Client --> Kong API Gateway (8080) --> Microservices (8001-8012)
```

### Port Mapping

- **8080**: Main API Gateway (proxy)
- **8001**: Kong Admin API
- **8002**: Kong Admin GUI
- **1337**: Konga Management UI

## Quick Start

### 1. Start Kong with Docker Compose

```bash
# Start all services including Kong
make up

# Or start Kong separately
docker-compose -f docker-compose.microservices.yml up -d kong
```

### 2. Setup Kong Configuration

```bash
# Automatic setup using script
make kong-setup

# Or manual setup
./services/api-gateway/kong-setup.sh

# Or using Python script
python3 services/api-gateway/kong-admin.py setup
```

### 3. Access Admin Interfaces

```bash
# Show admin URLs
make kong-admin

# Direct URLs:
# Kong Admin API: http://localhost:8001
# Kong Admin GUI: http://localhost:8002
# Konga UI: http://localhost:1337
```

## API Routes

All microservices are accessible through the Kong gateway at `http://localhost:8080`:

| Service   | Route              | Authentication | Rate Limit (min/hour) |
| --------- | ------------------ | -------------- | --------------------- |
| Auth      | `/auth/*`          | None           | 60/600                |
| Users     | `/users/*`         | JWT Required   | 100/1000              |
| Reminders | `/reminders/*`     | JWT Required   | 150/1500              |
| Notes     | `/notes/*`         | JWT Required   | 200/2000              |
| Expenses  | `/expenses/*`      | JWT Required   | 100/1000              |
| Friends   | `/friends/*`       | JWT Required   | 50/500                |
| Logs      | `/logs/*`          | JWT Required   | 50/500                |
| STT       | `/stt/*`           | JWT Required   | 30/300                |
| TTS       | `/tts/*`           | JWT Required   | 30/300                |
| Intent    | `/intent/*`        | JWT Required   | 100/1000              |
| Chat      | `/conversations/*` | JWT Required   | 200/2000              |
| Jobs      | `/jobs/*`          | JWT Required   | 50/500                |
| Health    | `/health`          | None           | 100/1000              |

## Authentication Flow

1. **Login**: `POST http://localhost:8080/auth/login`
2. **Get JWT Token**: Response contains `access_token`
3. **Authenticated Requests**: Include `Authorization: Bearer <token>` header

Example:

```bash
# Login
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token for protected routes
curl -X GET http://localhost:8080/users/profile \
  -H "Authorization: Bearer <your-jwt-token>"
```

## Kong Configuration

### Services Configuration

Each microservice is registered as a Kong service with:

- **Upstream URL**: Internal service address (e.g., `http://auth-service:8000`)
- **Timeout Settings**: 60s connect/read/write timeouts
- **Retry Logic**: 3 retries on failure

### Plugin Configuration

- **JWT**: Token validation for protected routes
- **CORS**: Cross-origin support for web clients
- **Rate Limiting**: Per-service traffic limits
- **Correlation ID**: Request tracking with X-Request-ID
- **File Logging**: Access logs for monitoring

## Management Commands

```bash
# Kong-specific commands
make kong-setup     # Setup Kong configuration
make kong-logs      # Follow Kong logs
make kong-restart   # Restart Kong
make kong-admin     # Show admin URLs
make konga-logs     # Follow Konga logs

# Health checks through gateway
curl http://localhost:8080/health
```

## Configuration Files

- **`docker-compose.kong.yml`**: Standalone Kong setup
- **`kong.yml`**: Declarative Kong configuration
- **`kong-setup.sh`**: Bash setup script
- **`kong-admin.py`**: Python management script
- **`Dockerfile`**: Custom Kong image

## Monitoring

### Health Checks

```bash
# Kong health
curl http://localhost:8001/status

# Service health through gateway
curl http://localhost:8080/health
```

### Admin API Examples

```bash
# List services
curl http://localhost:8001/services/

# List routes
curl http://localhost:8001/routes/

# List plugins
curl http://localhost:8001/plugins/

# Service metrics
curl http://localhost:8001/services/auth-service
```

### Logs

```bash
# Kong access logs
make kong-logs

# Konga admin logs
make konga-logs

# Service logs through gateway
docker-compose logs -f kong
```

## Troubleshooting

### Common Issues

1. **Kong Not Starting**

   ```bash
   # Check database connection
   docker-compose logs kong-database

   # Check migrations
   docker-compose logs kong-migration
   ```

2. **Service Not Reachable**

   ```bash
   # Verify service is running
   docker-compose ps

   # Check Kong configuration
   curl http://localhost:8001/services/service-name
   ```

3. **JWT Authentication Failing**

   ```bash
   # Check JWT consumer
   curl http://localhost:8001/consumers/eindr-auth-service

   # Verify JWT secret matches auth service
   ```

4. **Rate Limiting Issues**
   ```bash
   # Check rate limiting plugin
   curl http://localhost:8001/services/service-name/plugins
   ```

### Debug Mode

```bash
# Start Kong with debug logging
KONG_LOG_LEVEL=debug docker-compose up kong
```

## Production Considerations

1. **SSL/TLS**: Configure Kong with SSL certificates
2. **Database**: Use external PostgreSQL for Kong database
3. **Secrets**: Change default JWT secret and database passwords
4. **Monitoring**: Integrate with Prometheus/Grafana
5. **Caching**: Enable Kong caching plugins
6. **Load Balancing**: Configure upstream load balancing

## Security

- JWT tokens are validated at the gateway level
- Rate limiting prevents abuse
- CORS is configured for web client support
- Request logging for audit trails
- No direct service access from outside

## Performance

- Connection pooling to microservices
- Request/response buffering
- Automatic retries on failure
- Health checks for upstream services
- Efficient routing with minimal latency
