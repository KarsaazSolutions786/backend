# Coqui STT Integration for Eindr Backend

This document describes the Coqui STT (Speech-to-Text) integration in the Eindr backend API.

## Overview

The Eindr backend now includes offline speech-to-text transcription using the Coqui STT library. This allows users to upload WAV audio files and receive transcribed text without relying on external cloud services.

## Features

- **Offline Processing**: No internet connection required for transcription
- **Audio Format Validation**: Automatic validation of WAV file format requirements
- **Audio Preprocessing**: Automatic conversion of audio files to meet Coqui STT requirements
- **Intent Classification**: Optional integration with intent classification for transcribed text
- **Comprehensive Error Handling**: Detailed error messages for troubleshooting

## Audio Requirements

Coqui STT requires specific audio format specifications:

- **Format**: WAV (.wav)
- **Sample Rate**: 16 kHz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit PCM
- **Maximum File Size**: 10 MB (configurable)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The following packages are required for Coqui STT:
- `stt` - Coqui STT library
- `librosa` - Audio processing
- `soundfile` - Audio file I/O
- `pydub` - Audio format conversion

### 2. Download Coqui STT Model

Run the model download script:

```bash
python download_coqui_model.py
```

This will download:
- `models/coqui-stt.tflite` - The main STT model (~180 MB)
- `models/coqui-stt-scorer.scorer` - Language model scorer for improved accuracy (~950 MB)

### 3. Start the Server

```bash
python main.py
```

## API Endpoints

### POST `/api/v1/stt/transcribe`

Transcribe an uploaded WAV audio file to text.

**Headers:**
```
Authorization: Bearer <firebase_token>
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/stt/transcribe" \
  -H "Authorization: Bearer YOUR_FIREBASE_TOKEN" \
  -F "audio_file=@your_audio.wav"
```

**Response:**
```json
{
  "success": true,
  "transcription": "Hello, this is a test transcription",
  "intent": {
    "intent": "greeting",
    "confidence": 0.95
  },
  "user_id": "user123",
  "model_info": {
    "status": "loaded",
    "model_type": "coqui_stt",
    "model_path": "./models/coqui-stt.tflite",
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16
  },
  "audio_requirements": {
    "format": "WAV",
    "sample_rate": "16000Hz",
    "channels": "Mono",
    "bit_depth": "16-bit PCM"
  }
}
```

### GET `/api/v1/stt/model-info`

Get information about the loaded STT model and audio requirements.

**Headers:**
```
Authorization: Bearer <firebase_token>
```

**Response:**
```json
{
  "model_info": {
    "status": "loaded",
    "model_type": "coqui_stt",
    "model_path": "./models/coqui-stt.tflite",
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16
  },
  "audio_requirements": {
    "supported_formats": [".wav", ".wave"],
    "sample_rate": "16000Hz",
    "channels": 1,
    "bit_depth": "16-bit",
    "max_file_size": "10.0MB"
  },
  "preprocessing": {
    "automatic_resampling": true,
    "automatic_mono_conversion": true,
    "automatic_bit_depth_conversion": true
  }
}
```

## Audio Preprocessing

The system automatically handles audio files that don't meet the exact requirements:

1. **Resampling**: Converts any sample rate to 16 kHz
2. **Channel Conversion**: Converts stereo to mono
3. **Bit Depth Conversion**: Converts to 16-bit PCM format

This is done using the `librosa` library for robust audio processing.

## Testing

### 1. Run the Test Script

```bash
python test_transcription.py
```

This script will:
- Create a test WAV file with correct format
- Test the model info endpoint
- Test the transcription endpoint
- Validate the audio file format

### 2. Manual Testing with cURL

```bash
# Test model info
curl -X GET "http://localhost:8000/api/v1/stt/model-info" \
  -H "Authorization: Bearer YOUR_FIREBASE_TOKEN"

# Test transcription
curl -X POST "http://localhost:8000/api/v1/stt/transcribe" \
  -H "Authorization: Bearer YOUR_FIREBASE_TOKEN" \
  -F "audio_file=@test_audio.wav"
```

## Configuration

Update `core/config.py` to customize settings:

```python
class Settings(BaseModel):
    # Coqui STT model path
    COQUI_STT_MODEL_PATH: str = "./models/coqui-stt.tflite"
    
    # Audio settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_BIT_DEPTH: int = 16
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    SUPPORTED_AUDIO_FORMATS: List[str] = [".wav", ".wave"]
```

## Error Handling

The API provides detailed error messages for common issues:

### File Format Errors
```json
{
  "detail": "Unsupported audio format '.mp3'. Supported formats: .wav, .wave"
}
```

### Audio Format Validation Errors
```json
{
  "detail": "Transcription failed. Please ensure your audio file meets the requirements: WAV format, 16kHz sample rate, mono channel, 16-bit PCM"
}
```

### Service Availability Errors
```json
{
  "detail": "Speech-to-text service not available"
}
```

## Troubleshooting

### 1. Model Not Found
```
WARNING: Model file not found at ./models/coqui-stt.tflite
```
**Solution**: Run `python download_coqui_model.py` to download the model.

### 2. Package Import Error
```
WARNING: Coqui STT not available, running in demo mode
```
**Solution**: Install the STT package: `pip install stt`

### 3. Audio Format Issues
```
ERROR: Sample rate must be 16000Hz, got 44100Hz
```
**Solution**: The system will automatically preprocess the audio, but for best results, use the correct format.

### 4. Large File Upload
```
ERROR: File too large. Maximum size: 10.0MB
```
**Solution**: Reduce file size or increase `MAX_FILE_SIZE` in configuration.

## Performance Considerations

- **Model Loading**: The model is loaded once at startup and cached in memory
- **File Processing**: Temporary files are automatically cleaned up after processing
- **Memory Usage**: The model requires ~200MB of RAM when loaded
- **Processing Time**: Transcription typically takes 10-30% of the audio duration

## Demo Mode

If the Coqui STT model is not available, the service runs in demo mode:
- Returns random sample transcriptions
- Useful for development and testing
- No actual speech processing occurs

## Integration with Other Services

The transcription endpoint integrates with:
- **Intent Classification**: Automatically classifies the intent of transcribed text
- **Chat Service**: Can be used with the chat response generation
- **User Authentication**: Requires Firebase authentication

## Model Information

The default model used is:
- **Model**: Coqui STT English v1.0.0 (huge vocabulary)
- **Size**: ~180 MB (model) + ~950 MB (scorer)
- **Language**: English
- **Accuracy**: High accuracy for clear speech in quiet environments

For other languages or custom models, update the download URLs in `download_coqui_model.py`.

## Security Considerations

- All uploaded files are temporarily stored and automatically deleted
- Firebase authentication is required for all endpoints
- File size limits prevent abuse
- Only WAV files are accepted to prevent malicious uploads 