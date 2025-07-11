# AI Pipeline Service - Testing Guide

## 🚀 Quick Start Testing

### 1. Start the Service
```bash
# Option 1: Using Docker Compose (recommended)
cd /Users/afnan/Dev/microservices/backend
docker-compose up ai-pipeline-service

# Option 2: Direct service start
cd services/ai-pipeline-service
uvicorn src.main:app --reload --port 8007 --host 0.0.0.0
```

### 2. Verify Service is Running
```bash
curl http://localhost:8007/health
```
Expected response:
```json
{"status": "healthy", "service": "ai-pipeline-service"}
```

## 📝 Testing Endpoints

### Health Check
```bash
curl http://localhost:8007/health
```

### Text Processing
```bash
curl -X POST http://localhost:8007/pipeline/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "customer_id": "test_user_123",
    "multi_intent": true,
    "voice": "en-US-Standard-A"
  }'
```

### Audio Processing
```bash
curl -X POST http://localhost:8007/pipeline/transcribe-and-respond \
  -F "audio=@test_audio.wav" \
  -F "customer_id=test_user_123" \
  -F "voice=en-US-Standard-A"
```

## 🧪 Automated Testing

### Run the Test Suite
```bash
cd services/ai-pipeline-service
python test_pipeline.py
```

### Test Individual Functions
```python
import asyncio
from test_pipeline import AIPipelineTester

async def quick_test():
    async with AIPipelineTester() as tester:
        # Test health
        await tester.test_health()
        
        # Test text processing
        await tester.test_process_text(
            text="Hello world",
            customer_id="test123"
        )

asyncio.run(quick_test())
```

## 🔧 Prerequisites for Testing

### Required Services
The AI Pipeline Service depends on these services:
- **STT Service**: `http://localhost:8008`
- **Intent Service**: `http://localhost:8010`
- **Chat Service**: `http://localhost:8011`
- **TTS Service**: `http://localhost:8009`

### Start All Dependencies
```bash
# Start all AI services
cd /Users/afnan/Dev/microservices/backend
docker-compose up stt-service intent-service chat-service tts-service ai-pipeline-service
```

## 📊 Test Scenarios

### 1. Basic Text Processing
```bash
# Test with different intents
curl -X POST http://localhost:8007/pipeline/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Set a reminder for tomorrow", "customer_id": "user123"}'
```

### 2. Multi-Intent Processing
```bash
curl -X POST http://localhost:8007/pipeline/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello and set a reminder", "customer_id": "user123", "multi_intent": true}'
```

### 3. Text-to-Speech Integration
```bash
curl -X POST http://localhost:8007/pipeline/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this will be converted to speech",
    "customer_id": "user123",
    "voice": "en-US-Standard-A"
  }'
```

### 4. Full Audio Pipeline
```bash
# Create a test audio file first
echo "Hello, this is a test" | text2wave -o test_audio.wav

# Test the full pipeline
curl -X POST http://localhost:8007/pipeline/transcribe-and-respond \
  -F "audio=@test_audio.wav" \
  -F "customer_id=user123"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Service Not Responding
```bash
# Check if service is running
docker ps | grep ai-pipeline-service

# Check logs
docker logs backend-ai-pipeline-service-1
```

#### 2. Dependency Services Unavailable
```bash
# Check all AI services
curl http://localhost:8008/health  # STT
curl http://localhost:8010/health  # Intent
curl http://localhost:8011/health  # Chat
curl http://localhost:8009/health  # TTS
```

#### 3. Timeout Errors
```bash
# Increase timeout in config
export AI_HTTP_TIMEOUT=60
```

### Debug Mode
```bash
# Start with debug logging
cd services/ai-pipeline-service
uvicorn src.main:app --reload --port 8007 --log-level debug
```

## 📈 Performance Testing

### Load Testing
```python
import asyncio
import httpx
import time

async def load_test():
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        tasks = []
        
        for i in range(10):
            task = client.post(
                "http://localhost:8007/pipeline/process",
                json={"text": f"Test message {i}", "customer_id": "user123"}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"Processed {len(responses)} requests in {end_time - start_time:.2f} seconds")
        print(f"Average response time: {(end_time - start_time) / len(responses):.2f} seconds")

asyncio.run(load_test())
```

### Memory Usage Monitoring
```bash
# Monitor service resource usage
docker stats backend-ai-pipeline-service-1
```

## 🔍 Integration Testing

### Test with Other Services
```bash
# Test complete workflow
# 1. Create user
curl -X POST http://localhost:8001/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpass"}'

# 2. Get token
TOKEN=$(curl -X POST http://localhost:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpass"}' | jq -r '.access_token')

# 3. Test AI pipeline
curl -X POST http://localhost:8007/pipeline/process \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text": "Hello", "customer_id": "1"}'
```

## 📋 Test Checklist

- [ ] Service starts successfully
- [ ] Health endpoint responds
- [ ] Text processing works
- [ ] Audio processing works
- [ ] TTS integration works
- [ ] Error handling works
- [ ] Performance is acceptable
- [ ] Integration with other services works

## 🎯 Expected Responses

### Successful Text Processing
```json
{
  "reply_text": "Hello! How can I help you today?",
  "intent": "chit_chat",
  "audio_url": "https://example.com/audio/response.wav"
}
```

### Successful Audio Processing
```json
{
  "reply_text": "I heard you say: Hello, how are you?",
  "intent": "chit_chat",
  "audio_url": "https://example.com/audio/response.wav"
}
```

### Error Response
```json
{
  "detail": "Downstream AI service error: Connection timeout"
}
``` 