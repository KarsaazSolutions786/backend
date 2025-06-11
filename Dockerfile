# Multi-stage Dockerfile for Eindr Backend with optimized size
# Supports both minimal mode (Railway) and full AI mode (with runtime model loading)

# ====== Build Stage ======
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Build argument for minimal mode
ARG MINIMAL_MODE=false
ENV MINIMAL_MODE=${MINIMAL_MODE}

# Install only essential build dependencies and clean up in same layer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    pkg-config \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt requirements.railway.txt ./

# Create virtual environment and install dependencies in one layer
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    if [ "$MINIMAL_MODE" = "true" ]; then \
        echo "Installing minimal dependencies for Railway..."; \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.railway.txt; \
    else \
        echo "Installing full dependencies..."; \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt; \
        if command -v nvcc >/dev/null 2>&1; then \
            echo "GPU detected, installing vLLM with CUDA support..."; \
            pip install --no-cache-dir vllm==0.2.7; \
        else \
            echo "No GPU detected, skipping vLLM for smaller image..."; \
        fi; \
        pip install --no-cache-dir accelerate>=0.20.0; \
    fi && \
    pip cache purge && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true

# ====== Production Stage ======
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # PyTorch memory optimization
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    # Model configurations
    MODEL_DOWNLOAD_URL="" \
    HUGGINGFACE_HUB_CACHE=/app/models/cache

# Install minimal runtime dependencies and clean up in same layer
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    postgresql-client \
    libsndfile1 \
    ffmpeg \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only essential application code (models excluded via .dockerignore)
COPY . .

# Create directories and startup scripts in single layer
RUN mkdir -p models logs uploads scripts models/cache && \
    chmod -R 755 models && \
    chmod -R 777 logs uploads && \
    printf '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting Eindr Backend..."\n\
echo "MINIMAL_MODE: ${MINIMAL_MODE:-false}"\n\
echo "PORT: ${PORT:-8000}"\n\
\n\
# Function to download models if needed\n\
download_models() {\n\
    if [ "${MINIMAL_MODE:-false}" != "true" ] && [ ! -f "./models/Bloom560m.bin" ]; then\n\
        if [ -n "$MODEL_DOWNLOAD_URL" ]; then\n\
            echo "📥 Downloading Bloom-560M model..."\n\
            curl -L "$MODEL_DOWNLOAD_URL" -o ./models/Bloom560m.bin || echo "⚠️ Model download failed, continuing without Bloom-560M"\n\
        else\n\
            echo "ℹ️ MODEL_DOWNLOAD_URL not set, skipping Bloom-560M download"\n\
        fi\n\
    fi\n\
}\n\
\n\
# Function to start main FastAPI server\n\
start_main_server() {\n\
    echo "🌟 Starting main FastAPI server on port ${PORT}..."\n\
    exec python -m uvicorn main:app \\\n\
        --host 0.0.0.0 \\\n\
        --port ${PORT} \\\n\
        --timeout-keep-alive 30 \\\n\
        --workers 1 \\\n\
        --limit-concurrency 100 \\\n\
        --limit-max-requests 1000\n\
}\n\
\n\
# Download models if needed\n\
download_models\n\
\n\
if [ "${MINIMAL_MODE:-false}" = "true" ]; then\n\
    echo "🚀 Starting in MINIMAL MODE (Railway deployment)"\n\
    start_main_server\n\
else\n\
    echo "🚀 Starting in FULL MODE"\n\
    if [ -f "./models/Bloom560m.bin" ]; then\n\
        echo "✅ Bloom-560M model found, full AI features available"\n\
    else\n\
        echo "⚠️ Bloom-560M model not found, using fallback chat service"\n\
    fi\n\
    start_main_server\n\
fi\n' > /app/start_server.sh && \
    chmod +x /app/start_server.sh && \
    printf '#!/bin/bash\n\
# Lightweight health check\n\
if ! curl -f http://localhost:${PORT}/health >/dev/null 2>&1; then\n\
    echo "Health check failed"\n\
    exit 1\n\
fi\n\
echo "Health check passed"\n\
exit 0\n' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + || true

# Expose port
EXPOSE ${PORT}

# Lightweight health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Use startup script as entrypoint
CMD ["/app/start_server.sh"] 