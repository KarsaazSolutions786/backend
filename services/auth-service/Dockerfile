FROM python:3.11-slim

# Add build argument to force rebuild
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# Set the working directory
WORKDIR /app

# Updated Dockerfile - no start.sh dependency

# Install system dependencies in one layer to reduce size
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with timeout and no cache
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt


COPY start.sh .
RUN chmod +x start.sh

# Copy source code
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application using the run script
CMD ["python", "-u", "run.py"] 