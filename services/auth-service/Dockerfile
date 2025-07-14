FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ./services/auth-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./services/auth-service/src/ ./src/
COPY ./shared/ ./shared/
COPY ./services/auth-service/migrations/ ./migrations/
COPY ./services/auth-service/alembic.ini .

# Start the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 