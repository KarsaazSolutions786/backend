# Multi-stage Dockerfile for Eindr Backend with Bloom-560M support
# Supports both minimal mode (Railway) and full AI mode (with vLLM)

# ====== Build Stage ======
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Build argument for minimal mode
ARG MINIMAL_MODE=false
ENV MINIMAL_MODE=${MINIMAL_MODE}

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    pkg-config \
    libsndfile1-dev \
    ffmpeg \
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
RUN printf '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Eindr Backend..."\n\
echo "MINIMAL_MODE: ${MINIMAL_MODE:-false}"\n\
echo "PORT: ${PORT:-8000}"\n\
\n\
# Function to start main FastAPI server\n\
start_main_server() {\n\
    echo "Starting main FastAPI server on port ${PORT}..."\n\
    exec python -m uvicorn main:app \\\n\
        --host 0.0.0.0 \\\n\
        --port ${PORT} \\\n\
        --timeout-keep-alive 30 \\\n\
        --workers 1 \\\n\
        --limit-concurrency 100 \\\n\
        --limit-max-requests 1000\n\
}\n\
\n\
# Function to start vLLM server for Bloom-560M\n\
start_vllm_server() {\n\
    if [ -f "./models/Bloom560m.bin" ]; then\n\
        echo "Starting vLLM server for Bloom-560M on port ${VLLM_SERVER_PORT}..."\n\
        python -m vllm.entrypoints.openai.api_server \\\n\
            --model ./models/Bloom560m.bin \\\n\
            --host 0.0.0.0 \\\n\
            --port ${VLLM_SERVER_PORT} \\\n\
            --max-model-len ${VLLM_MAX_MODEL_LEN:-2048} \\\n\
            --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.8} \\\n\
            --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE:-1} \\\n\
            --disable-log-requests \\\n\
            --served-model-name bigscience/bloom-560m &\n\
        \n\
        VLLM_PID=$!\n\
        echo "vLLM server started with PID: $VLLM_PID"\n\
        \n\
        # Wait for vLLM to be ready\n\
        echo "Waiting for vLLM server to be ready..."\n\
        for i in {1..30}; do\n\
            if curl -s http://localhost:${VLLM_SERVER_PORT}/health > /dev/null 2>&1; then\n\
                echo "✅ vLLM server is ready!"\n\
                break\n\
            fi\n\
            echo "Waiting for vLLM server... ($i/30)"\n\
            sleep 5\n\
        done\n\
        \n\
        # Store PID for cleanup\n\
        echo $VLLM_PID > /tmp/vllm.pid\n\
    else\n\
        echo "⚠️  Bloom560m.bin not found, skipping vLLM server"\n\
    fi\n\
}\n\
\n\
# Function to cleanup on exit\n\
cleanup() {\n\
    echo "Cleaning up..."\n\
    if [ -f /tmp/vllm.pid ]; then\n\
        VLLM_PID=$(cat /tmp/vllm.pid)\n\
        if kill -0 $VLLM_PID 2>/dev/null; then\n\
            echo "Stopping vLLM server (PID: $VLLM_PID)..."\n\
            kill $VLLM_PID\n\
        fi\n\
        rm -f /tmp/vllm.pid\n\
    fi\n\
    exit 0\n\
}\n\
\n\
# Set up signal handlers\n\
trap cleanup SIGTERM SIGINT\n\
\n\
if [ "${MINIMAL_MODE:-false}" = "true" ]; then\n\
    echo "🚀 Starting in MINIMAL MODE (Railway deployment)"\n\
    start_main_server\n\
else\n\
    echo "🚀 Starting in FULL MODE with Bloom-560M"\n\
    \n\
    # Start vLLM server in background\n\
    start_vllm_server\n\
    \n\
    # Start main server in foreground\n\
    start_main_server\n\
fi\n' > /app/start_server.sh

# Make startup script executable
RUN chmod +x /app/start_server.sh

# Create healthcheck script
RUN printf '#!/bin/bash\n\
# Health check for both minimal and full modes\n\
\n\
# Check main FastAPI server\n\
if ! curl -f http://localhost:${PORT}/health >/dev/null 2>&1; then\n\
    echo "Main server health check failed"\n\
    exit 1\n\
fi\n\
\n\
# In full mode, also check vLLM server\n\
if [ "${MINIMAL_MODE:-false}" != "true" ]; then\n\
    if [ -f "./models/Bloom560m.bin" ]; then\n\
        if ! curl -f http://localhost:${VLLM_SERVER_PORT}/health >/dev/null 2>&1; then\n\
            echo "vLLM server health check failed"\n\
            exit 1\n\
        fi\n\
    fi\n\
fi\n\
\n\
echo "Health check passed"\n\
exit 0\n' > /app/healthcheck.sh

RUN chmod +x /app/healthcheck.sh

# Expose ports
EXPOSE ${PORT}
EXPOSE ${VLLM_SERVER_PORT}

# Health check that works with both modes
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD /app/healthcheck.sh

# Use startup script as entrypoint
CMD ["/app/start_server.sh"] 