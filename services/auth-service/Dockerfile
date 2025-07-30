FROM python:3.11-slim

# Add build argument to force rebuild
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# Set default port
ENV PORT=8000

# Ensure the working directory is set to /app
WORKDIR /app
# Set environment variables
ENV PYTHONPATH=/app:/app/src:/app/shared
ENV PYTHONUNBUFFERED=1
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements first for better caching
COPY backend/services/auth-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and shared dependencies
COPY backend/services/auth-service/src/ ./src/
COPY backend/shared/ ./shared/
COPY backend/services/auth-service/migrations/ ./migrations/
COPY backend/services/auth-service/alembic.ini .



# Expose the port
EXPOSE $PORT

# Copy and make executable
COPY backend/services/auth-service/start.sh ./start.sh
RUN chmod +x /app/start.sh

# Run with environment variables
CMD ["/app/start.sh"]
