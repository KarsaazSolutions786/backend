# Running Bloom-560M Conversational AI

This guide explains how to set up and run the Bloom-560M conversational AI integration for handling `chit_chat` and `general_query` intents in the Eindr backend.

## 🎯 Overview

The Bloom-560M integration provides sophisticated conversational AI capabilities for:
- **Chit-chat conversations**: Natural dialogue and general questions
- **Task completion acknowledgments**: Contextual responses after successful operations
- **Help and guidance**: Intelligent assistance and suggestions
- **Conversation history**: Multi-turn dialogue with memory

## 🏗️ Architecture

```
Voice Input → STT → Intent Classification → Decision Router
                                               ↓
    Chit-Chat/General Query → ChatService → Bloom-560M via vLLM → Response
                                  ↓
                            Database (History) → TTS → Audio Response
```

**Key Components:**
- **ChatService**: Manages Bloom-560M interactions and conversation context
- **vLLM Server**: High-performance inference engine for Bloom-560M
- **Fallback Mode**: Enhanced contextual responses when model unavailable

## 🚀 Quick Start

### 1. Minimal Mode (Railway/Production)
For deployment environments with limited resources:

```bash
# Set environment variable
export MINIMAL_MODE=true

# Start the server
docker run -e MINIMAL_MODE=true your-app:latest
```

**Features in Minimal Mode:**
- ✅ Enhanced contextual responses
- ✅ Task completion acknowledgments  
- ✅ Conversation history tracking
- ❌ Full Bloom-560M AI model

### 2. Full Mode (Local Development)
For full AI capabilities with Bloom-560M:

```bash
# Ensure model is available
ls -la models/Bloom560m.bin

# Start with vLLM
export MINIMAL_MODE=false
docker run -p 8000:8000 -p 8001:8001 your-app:latest
```

## 📋 Prerequisites

### For Full Mode:
- **GPU**: Recommended for optimal performance (8GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Model File**: `models/Bloom560m.bin` (1GB)
- **vLLM**: Automatically installed in Docker

### For Minimal Mode:
- **RAM**: 2GB+ system memory
- **No GPU required**
- **No model file needed**

## ⚙️ Configuration

### Environment Variables

```bash
# === Core Settings ===
MINIMAL_MODE=false                    # Enable/disable full AI mode

# === vLLM Server Configuration ===
VLLM_SERVER_URL=http://localhost:8001 # vLLM server endpoint
VLLM_SERVER_PORT=8001                 # Port for vLLM server
VLLM_GPU_MEMORY_UTILIZATION=0.8       # GPU memory usage (0.0-1.0)
VLLM_MAX_MODEL_LEN=2048               # Maximum sequence length
VLLM_TENSOR_PARALLEL_SIZE=1           # Multi-GPU parallelization

# === Chat Generation Parameters ===
CHAT_MODEL_NAME=bigscience/bloom-560m # Model identifier
CHAT_MAX_TOKENS=150                   # Max response length
CHAT_TEMPERATURE=0.7                  # Randomness (0.0-1.0)
CHAT_TOP_P=0.9                       # Nucleus sampling
CHAT_TOP_K=50                         # Top-K sampling
CHAT_REPETITION_PENALTY=1.1           # Reduce repetition
```

### Docker Environment

```dockerfile
# Minimal mode (Railway)
ENV MINIMAL_MODE=true

# Full mode (Local/GPU)
ENV MINIMAL_MODE=false
ENV VLLM_GPU_MEMORY_UTILIZATION=0.8
ENV CHAT_TEMPERATURE=0.7
```

## 🔧 Installation & Setup

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone <your-repo>
cd eindr-backend

# Build image
docker build -t eindr-backend .

# Run in minimal mode
docker run -p 8000:8000 \
  -e MINIMAL_MODE=true \
  eindr-backend

# Run in full mode (with GPU)
docker run --gpus all -p 8000:8000 -p 8001:8001 \
  -e MINIMAL_MODE=false \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.8 \
  eindr-backend
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install vllm==0.2.7

# Start vLLM server (separate terminal)
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Bloom560m.bin \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8

# Start main server
export MINIMAL_MODE=false
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing the Integration

### 1. Health Check

```bash
# Check overall system health
curl http://localhost:8000/health

# Check chat service specifically
curl http://localhost:8000/api/chat/health
```

### 2. Test Chat Functionality

```bash
# Test via API
curl -X POST http://localhost:8000/api/stt/transcribe-and-respond \
  -F "audio_file=@test_audio.wav" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. Direct Chat Service Test

```python
import asyncio
from services.chat_service import ChatService

async def test_chat():
    service = ChatService()
    
    # Test basic conversation
    response = await service.generate_response(
        message="Hello, how can you help me?",
        user_id="test_user"
    )
    print(f"Response: {response}")
    
    # Test with task completion context
    context = {
        "intent": "create_reminder",
        "confidence": 0.9,
        "processing_result": {
            "success": True,
            "data": {"title": "doctor appointment", "time": "3 PM"}
        }
    }
    
    response = await service.generate_response(
        message="remind me about doctor",
        user_id="test_user",
        context=context
    )
    print(f"Contextual response: {response}")

# Run test
asyncio.run(test_chat())
```

### 4. Unit Tests

```bash
# Run ChatService tests
pytest tests/test_chat_service.py -v

# Run specific test categories
pytest tests/test_chat_service.py::TestChatService::test_generate_response_minimal_mode -v
```

## 🎛️ Service Configuration

### ChatService Integration Points

The ChatService is automatically integrated into the AI pipeline:

1. **Intent Classification** → Routes `chit_chat`/`general_query` to ChatService
2. **Context Processing** → Passes task completion status to chat generation
3. **Response Generation** → Creates contextual, personality-driven responses
4. **History Management** → Maintains conversation context across interactions
5. **TTS Integration** → Converts chat responses to speech

### Conversation Context

The ChatService maintains rich context:

```python
context = {
    "intent": "chit_chat",
    "confidence": 0.85,
    "original_transcription": "hello how are you",
    "processing_result": {
        "success": True,
        "data": {"title": "meeting", "time": "2 PM"}
    }
}
```

This enables responses like:
> "Perfect! I've created your reminder for 'meeting' at 2 PM. I'll make sure to notify you right on time. Is there anything else you'd like me to help you organize?"

## 📊 Monitoring & Metrics

### Service Status Endpoints

```bash
# Chat service health
GET /api/chat/health

# Model information
GET /api/chat/model-info

# Conversation summary
GET /api/chat/summary/{user_id}

# Clear conversation history
DELETE /api/chat/history/{user_id}
```

### Performance Metrics

Monitor these key metrics:

- **Response Time**: Target <2s for chat responses
- **Model Availability**: vLLM server uptime
- **Fallback Rate**: Percentage using enhanced fallback vs full AI
- **Conversation Quality**: User engagement and satisfaction

### Logs and Debugging

```bash
# Check service logs
docker logs <container_id> | grep "ChatService"

# vLLM server logs
docker logs <container_id> | grep "vLLM"

# Pipeline routing logs
docker logs <container_id> | grep "chit_chat\|general_query"
```

## 🔍 Troubleshooting

### Common Issues

**1. vLLM Server Not Starting**
```bash
# Check GPU availability
nvidia-smi

# Check model file
ls -la models/Bloom560m.bin

# Reduce memory utilization
export VLLM_GPU_MEMORY_UTILIZATION=0.6
```

**2. Out of Memory Errors**
```bash
# Reduce model length
export VLLM_MAX_MODEL_LEN=1024

# Use CPU inference (slower)
export CUDA_VISIBLE_DEVICES=""
```

**3. Chat Service Not Responding**
```bash
# Check service initialization
curl http://localhost:8000/api/chat/health

# Verify intent routing
grep -i "chit_chat" logs/app.log
```

**4. Poor Response Quality**
```bash
# Adjust generation parameters
export CHAT_TEMPERATURE=0.5        # More focused
export CHAT_REPETITION_PENALTY=1.2 # Less repetitive
export CHAT_MAX_TOKENS=200          # Longer responses
```

### Fallback Scenarios

The system automatically falls back in these cases:
- vLLM server unavailable → Enhanced contextual responses
- GPU memory exhausted → CPU inference or minimal mode
- Model loading failed → Rule-based intelligent responses
- Network timeout → Cached response patterns

## 🚀 Production Deployment

### Railway (Minimal Mode)
```bash
# Deploy with minimal mode
railway up --environment production
```

### Self-Hosted (Full Mode)
```bash
# Docker Compose setup
version: '3.8'
services:
  eindr-backend:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - MINIMAL_MODE=false
      - VLLM_GPU_MEMORY_UTILIZATION=0.8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Performance Optimization

**For GPU Deployments:**
- Use tensor parallelization for multiple GPUs
- Enable mixed precision inference
- Optimize batch sizes for throughput

**For CPU Deployments:**
- Use quantized models when available
- Increase worker processes
- Implement response caching

## 📈 Scaling Considerations

### Horizontal Scaling
- Deploy vLLM servers on dedicated GPU nodes
- Use load balancer for chat service instances
- Implement Redis for shared conversation history

### Model Optimization
- Consider model quantization (INT8/INT4)
- Implement response caching for common queries
- Use model distillation for smaller variants

## 🔐 Security & Privacy

### Data Handling
- Conversation history encrypted at rest
- PII detection and masking in responses
- Configurable data retention policies

### Access Control
- JWT token validation for all chat endpoints
- Rate limiting on chat requests
- Audit logging for conversation access

## 📚 API Reference

### Chat Endpoints

```bash
# Generate chat response
POST /api/chat/respond
{
  "message": "Hello, how are you?",
  "context": {...}
}

# Get conversation history
GET /api/chat/history/{user_id}

# Clear conversation history  
DELETE /api/chat/history/{user_id}

# Service health check
GET /api/chat/health
```

### Response Format

```json
{
  "success": true,
  "response": "Hello! I'm doing well, thank you. How can I help you stay organized today?",
  "context_used": true,
  "model_info": {
    "model_name": "bigscience/bloom-560m",
    "inference_engine": "vLLM",
    "response_time": 1.2
  }
}
```

---

## 🎉 Next Steps

After setting up Bloom-560M:

1. **Test Conversation Flow**: Try various chat scenarios
2. **Monitor Performance**: Check response times and quality
3. **Tune Parameters**: Adjust temperature and other settings
4. **Scale as Needed**: Add GPU resources or optimize for CPU
5. **Integrate Frontend**: Connect with your chat interface

For additional support, check the main README.md or raise an issue in the repository. 