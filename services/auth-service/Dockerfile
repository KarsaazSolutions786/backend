FROM python:3.11-slim

# Add build argument to force rebuild
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# Set default port
ENV PORT=8000

# Ensure the working directory is set to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dependencies
COPY services/auth-service/src/ ./src/
COPY shared/ ./shared/
COPY services/auth-service/migrations/ ./migrations/
COPY services/auth-service/alembic.ini .

# Set environment variables
ENV PYTHONPATH=/app:/app/src:/app/shared
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE $PORT

# Copy and make executable
COPY services/auth-service/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run with environment variables
CMD ["/app/start.sh"]
