FROM python:3.11-slim

# Add build argument to force rebuild
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# Set default port and ensure it's available
ENV PORT=8000
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

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

# Copy source code and shared dependencies
COPY src/ ./src/
COPY ../../shared/ ./shared/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Expose the port
EXPOSE $PORT

# Copy and make executable
COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Debug: Print environment before running
RUN echo "Build-time PORT: $PORT"

# Run with environment variables
CMD ["/app/start.sh"]
