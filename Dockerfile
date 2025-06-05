# Use Python 3.11 slim image for production
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MINIMAL_MODE=true \
    PORT=8000 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # PyTorch memory optimization
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

# Install system dependencies (minimal for Railway)
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    postgresql-client \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.railway.txt requirements.txt

# Install Python dependencies in stages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt || \
    (echo "Failed to install requirements" && exit 1)

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads && \
    chmod -R 777 logs uploads

# Expose port (Railway will set PORT environment variable)
EXPOSE ${PORT}

# Health check that works with minimal mode
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with proper signal handling and memory limits
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 30 --workers 1 --limit-concurrency 100 --limit-max-requests 1000"] 