# 🚀 AI Models Integration Setup Guide

## Overview
This guide will help you set up and integrate the three AI models (Whisper STT, MiniLM Intent Classification, and Coqui TTS) into your FastAPI backend for complete voice-to-database-to-speech workflow.

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space for models
- **GPU**: Optional but recommended for faster processing

### Audio Requirements
- **Input Format**: WAV, MP3, M4A, FLAC
- **Recommended**: 16kHz, Mono, 16-bit PCM WAV files
- **Max Duration**: 5 minutes per audio file
- **Min Duration**: 0.1 seconds

---

## 🔧 Installation Steps

### 1. Install AI Model Dependencies

```bash
# Install AI model requirements
pip install -r requirements_ai_models.txt

# Install additional audio dependencies (if needed)
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1 portaudio19-dev

# On macOS:
brew install ffmpeg portaudio

# On Windows:
# Download FFmpeg from https://ffmpeg.org/download.html
```

### 2. Download/Verify Model Files

Your model files should be placed in the `models/` directory:

```
models/
├── whisper-tiny.bin          # Whisper STT model (144MB)
├── Mini_LM.bin              # Fine-tuned MiniLM model (87MB)  
├── coqui.tflite             # Coqui TTS model (45MB)
└── models.py                # Model utilities
```

### 3. Environment Configuration

Add these settings to your environment variables or `.env` file:

```env
# AI Model Paths
WHISPER_MODEL_PATH=models/whisper-tiny.bin
MINILM_MODEL_PATH=models/Mini_LM.bin
COQUI_MODEL_PATH=models/coqui.tflite

# Device Configuration
AI_DEVICE=cuda  # or 'cpu' if no GPU available

# Audio Processing
MAX_AUDIO_SIZE_MB=100
SUPPORTED_AUDIO_FORMATS=wav,mp3,m4a,flac

# Pipeline Settings
ENABLE_MULTI_INTENT=true
DEFAULT_LANGUAGE=en
DEFAULT_TTS_VOICE=default
```

### 4. Update FastAPI Configuration

The new AI pipeline has already been integrated into your FastAPI app. Verify the following router is included in `main.py`:

```python
from api import ai_pipeline
app.include_router(ai_pipeline.router, prefix="/api/v1/ai-pipeline", tags=["AI Pipeline"])
```

---

## 🎯 API Endpoints

### Complete AI Pipeline
**POST** `/api/v1/ai-pipeline/complete-pipeline`

Processes audio through the entire pipeline: STT → Intent Classification → Database → TTS

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/ai-pipeline/complete-pipeline" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -F "audio_file=@recording.wav" \
  -F "language=en" \
  -F "multi_intent=true" \
  -F "store_in_database=true" \
  -F "generate_audio_response=true" \
  -F "voice=default"
```

**Response:**
```json
{
  "success": true,
  "pipeline_completed": true,
  "processing_time": 2.35,
  "transcription": "remind me to call John at 5 PM",
  "transcription_confidence": 0.95,
  "intent_result": {
    "intent": "create_reminder",
    "confidence": 0.92,
    "entities": {"person": ["John"], "time": ["5 PM"]}
  },
  "database_result": {
    "success": true,
    "table": "reminders",
    "reminder_id": "reminder_123"
  },
  "tts_result": {
    "success": true,
    "response_text": "Perfect! I've created a reminder to call John at 5 PM.",
    "audio_size": 45612,
    "audio_available": true
  }
}
```

### Individual Components

#### Transcription Only
**POST** `/api/v1/ai-pipeline/transcribe-only`

#### Intent Classification Only  
**POST** `/api/v1/ai-pipeline/classify-text-intent`

#### TTS Generation Only
**POST** `/api/v1/ai-pipeline/generate-speech`

#### Service Status
**GET** `/api/v1/ai-pipeline/service-status`

---

## 🔄 Workflow Examples

### 1. Voice Reminder Creation

```bash
# Upload audio: "Remind me to call Sarah tomorrow at 3 PM"
curl -X POST "http://localhost:8000/api/v1/ai-pipeline/complete-pipeline" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "audio_file=@reminder.wav"
```

**Pipeline Flow:**
1. **Whisper STT**: "remind me to call Sarah tomorrow at 3 PM"
2. **MiniLM Intent**: `create_reminder` (confidence: 0.94)
3. **Entity Extraction**: person="Sarah", time="tomorrow at 3 PM"
4. **Database**: New reminder created in `reminders` table
5. **Coqui TTS**: "Perfect! I've created a reminder to call Sarah tomorrow at 3 PM."

### 2. Multi-Intent Processing

```bash
# Upload audio: "Remind me to call John and note that he owes me $20"
curl -X POST "http://localhost:8000/api/v1/ai-pipeline/complete-pipeline" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "audio_file=@multi_intent.wav" \
  -F "multi_intent=true"
```

**Pipeline Flow:**
1. **Whisper STT**: "remind me to call John and note that he owes me $20"
2. **MiniLM Intent**: Multiple intents detected
   - `create_reminder`: "remind me to call John" 
   - `create_ledger`: "he owes me $20"
3. **Database**: 
   - Reminder created in `reminders` table
   - Ledger entry created in `ledger_entries` table
4. **Coqui TTS**: "Excellent! I've completed both tasks: created reminder to call John and recorded $20 with John."

### 3. Note Creation

```bash
# Upload audio: "Create a note about tomorrow's team meeting"
curl -X POST "http://localhost:8000/api/v1/ai-pipeline/complete-pipeline" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "audio_file=@note.wav"
```

**Pipeline Flow:**
1. **Whisper STT**: "create a note about tomorrow's team meeting"
2. **MiniLM Intent**: `create_note` (confidence: 0.91)
3. **Entity Extraction**: content="tomorrow's team meeting"
4. **Database**: New note created in `notes` table
5. **Coqui TTS**: "Great! I've saved your note about 'tomorrow's team meeting'."

---

## 🧪 Testing

### Run Unit Tests

```bash
# Run all AI pipeline tests
python test_ai_pipeline.py

# Run specific test categories
pytest test_ai_pipeline.py::TestWhisperSTTService -v
pytest test_ai_pipeline.py::TestMiniLMIntentService -v
pytest test_ai_pipeline.py::TestCoquiTTSService -v
pytest test_ai_pipeline.py::TestAIPipelineService -v
```

### Manual Testing

1. **Check Service Status:**
```bash
curl -X GET "http://localhost:8000/api/v1/ai-pipeline/service-status" \
  -H "Authorization: Bearer $JWT_TOKEN"
```

2. **Test Individual Components:**
```bash
# Test STT only
curl -X POST "http://localhost:8000/api/v1/ai-pipeline/transcribe-only" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "audio_file=@test.wav"

# Test Intent Classification
curl -X POST "http://localhost:8000/api/v1/ai-pipeline/classify-text-intent" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "remind me to call John", "multi_intent": true}'

# Test TTS
curl -X POST "http://localhost:8000/api/v1/ai-pipeline/generate-speech" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test message", "voice": "default"}'
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Model Loading Failures

**Error**: `Whisper model not found`
**Solution**: 
```bash
# Verify model file exists
ls -la models/whisper-tiny.bin

# If missing, download Whisper model
python -c "import whisper; whisper.load_model('tiny')"
```

#### 2. CUDA/GPU Issues

**Error**: `CUDA out of memory`
**Solution**:
```bash
# Force CPU usage
export AI_DEVICE=cpu

# Or reduce batch sizes in model configurations
```

#### 3. Audio Format Issues

**Error**: `Invalid audio format`
**Solution**:
```bash
# Convert audio to supported format
ffmpeg -i input.mp4 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

#### 4. Dependencies Missing

**Error**: `ModuleNotFoundError: No module named 'transformers'`
**Solution**:
```bash
# Reinstall requirements
pip install -r requirements_ai_models.txt --force-reinstall
```

### Performance Optimization

#### 1. GPU Acceleration
```bash
# Install CUDA-enabled PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Model Caching
```python
# Enable model caching in production
os.environ['TRANSFORMERS_CACHE'] = './model_cache/'
os.environ['HF_HOME'] = './model_cache/'
```

#### 3. Concurrent Processing
```python
# Adjust worker processes for better throughput
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

---

## 📊 Monitoring & Logging

### Pipeline Statistics

Access pipeline performance metrics:

```bash
curl -X GET "http://localhost:8000/api/v1/ai-pipeline/service-status" \
  -H "Authorization: Bearer $JWT_TOKEN"
```

**Response includes:**
- Total requests processed
- Success/failure rates
- Average processing times
- Model availability status

### Logs

Monitor AI pipeline logs:

```bash
# View recent logs
tail -f logs/app.log | grep "AI Pipeline"

# Filter specific components
tail -f logs/app.log | grep "Whisper\|MiniLM\|Coqui"
```

---

## 🔒 Security Considerations

### Authentication
- All endpoints require Firebase JWT authentication
- User isolation enforced at database level
- Audio files processed in isolated temporary directories

### Data Privacy
- Audio files automatically deleted after processing
- Transcriptions stored securely with user association
- No audio data retained beyond processing pipeline

### Rate Limiting
```python
# Implement rate limiting for AI endpoints
@limiter.limit("10/minute")
async def complete_ai_pipeline(...):
    pass
```

---

## 🚀 Production Deployment

### Docker Setup

```dockerfile
# Add to Dockerfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev

COPY requirements_ai_models.txt .
RUN pip install -r requirements_ai_models.txt

COPY models/ models/
```

### Environment Variables

```yaml
# docker-compose.yml
environment:
  - AI_DEVICE=cuda
  - WHISPER_MODEL_PATH=models/whisper-tiny.bin
  - MINILM_MODEL_PATH=models/Mini_LM.bin
  - COQUI_MODEL_PATH=models/coqui.tflite
```

---

## 📈 Scaling Considerations

### Horizontal Scaling
- Models loaded once per worker process
- Stateless design allows multiple replicas
- Consider model serving with TensorFlow Serving for large deployments

### Performance Metrics
- **STT Processing**: ~2-5 seconds per minute of audio
- **Intent Classification**: ~100ms per request
- **TTS Generation**: ~1-3 seconds per response
- **Total Pipeline**: ~3-8 seconds end-to-end

---

## 📚 Additional Resources

### Model Documentation
- [Whisper Documentation](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Coqui TTS](https://github.com/coqui-ai/TTS)

### API Examples
- See `SPEECH_INTENT_WORKFLOW_README.md` for detailed API workflows
- Check `test_ai_pipeline.py` for comprehensive test examples

### Support
- Check logs for detailed error messages
- Enable debug logging for development
- Monitor resource usage during processing

---

**🎉 Your AI pipeline is now ready! Start by testing with simple voice commands and gradually work up to complex multi-intent scenarios.** 