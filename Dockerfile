# Multi-stage Dockerfile for Eindr Backend with Bloom-560M support
# Supports both minimal mode (Railway) and full AI mode (with vLLM)

# ====== Build Stage ======
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libsndfile1-dev \
    libffi-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements.railway.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies based on mode
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    if [ "$MINIMAL_MODE" = "true" ]; then \
        echo "Installing minimal dependencies for Railway..."; \
        pip install --no-cache-dir -r requirements.railway.txt; \
    else \
        echo "Installing full dependencies including vLLM..."; \
        pip install --no-cache-dir -r requirements.txt && \
        pip install --no-cache-dir vllm==0.2.7 && \
        pip install --no-cache-dir accelerate>=0.20.0; \
    fi

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
    # vLLM configurations
    VLLM_SERVER_PORT=8001 \
    VLLM_SERVER_URL=http://localhost:8001

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    postgresql-client \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY . .

# Ensure models directory exists and has correct permissions
# Bloom560m.bin is tracked with Git LFS
RUN mkdir -p models && \
    chmod -R 755 models

# Create necessary directories
RUN mkdir -p logs uploads scripts && \
    chmod -R 777 logs uploads

# Create startup script for handling both minimal and full modes
RUN cat > /app/start_server.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting Eindr Backend..."
echo "MINIMAL_MODE: ${MINIMAL_MODE:-false}"
echo "PORT: ${PORT:-8000}"

# Function to start main FastAPI server
start_main_server() {
    echo "Starting main FastAPI server on port ${PORT}..."
    exec python -m uvicorn main:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --timeout-keep-alive 30 \
        --workers 1 \
        --limit-concurrency 100 \
        --limit-max-requests 1000
}

# Function to start vLLM server for Bloom-560M
start_vllm_server() {
    if [ -f "./models/Bloom560m.bin" ]; then
        echo "Starting vLLM server for Bloom-560M on port ${VLLM_SERVER_PORT}..."
        python -m vllm.entrypoints.openai.api_server \
            --model ./models/Bloom560m.bin \
            --host 0.0.0.0 \
            --port ${VLLM_SERVER_PORT} \
            --max-model-len ${VLLM_MAX_MODEL_LEN:-2048} \
            --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.8} \
            --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE:-1} \
            --disable-log-requests \
            --served-model-name bigscience/bloom-560m &
        
        VLLM_PID=$!
        echo "vLLM server started with PID: $VLLM_PID"
        
        # Wait for vLLM to be ready
        echo "Waiting for vLLM server to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:${VLLM_SERVER_PORT}/health > /dev/null 2>&1; then
                echo "✅ vLLM server is ready!"
                break
            fi
            echo "Waiting for vLLM server... ($i/30)"
            sleep 5
        done
        
        # Store PID for cleanup
        echo $VLLM_PID > /tmp/vllm.pid
    else
        echo "⚠️  Bloom560m.bin not found, skipping vLLM server"
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ -f /tmp/vllm.pid ]; then
        VLLM_PID=$(cat /tmp/vllm.pid)
        if kill -0 $VLLM_PID 2>/dev/null; then
            echo "Stopping vLLM server (PID: $VLLM_PID)..."
            kill $VLLM_PID
        fi
        rm -f /tmp/vllm.pid
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

if [ "${MINIMAL_MODE:-false}" = "true" ]; then
    echo "🚀 Starting in MINIMAL MODE (Railway deployment)"
    start_main_server
else
    echo "🚀 Starting in FULL MODE with Bloom-560M"
    
    # Start vLLM server in background
    start_vllm_server
    
    # Start main server in foreground
    start_main_server
fi
EOF

# Make startup script executable
RUN chmod +x /app/start_server.sh

# Create healthcheck script
RUN cat > /app/healthcheck.sh << 'EOF'
#!/bin/bash
# Health check for both minimal and full modes

# Check main FastAPI server
if ! curl -f http://localhost:${PORT}/health >/dev/null 2>&1; then
    echo "Main server health check failed"
    exit 1
fi

# In full mode, also check vLLM server
if [ "${MINIMAL_MODE:-false}" != "true" ]; then
    if [ -f "./models/Bloom560m.bin" ]; then
        if ! curl -f http://localhost:${VLLM_SERVER_PORT}/health >/dev/null 2>&1; then
            echo "vLLM server health check failed"
            exit 1
        fi
    fi
fi

echo "Health check passed"
exit 0
EOF

RUN chmod +x /app/healthcheck.sh

# Expose ports
EXPOSE ${PORT}
EXPOSE ${VLLM_SERVER_PORT}

# Health check that works with both modes
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD /app/healthcheck.sh

# Use startup script as entrypoint
CMD ["/app/start_server.sh"] 