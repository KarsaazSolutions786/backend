FROM python:3.11-slim

# Add build argument to force rebuild
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

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

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE ${PORT:-8000}
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
