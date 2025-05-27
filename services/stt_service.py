import json
import wave
import numpy as np
from typing import Optional
from core.config import settings
from utils.logger import logger

# Optional import for demo purposes
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("Vosk not available, running in demo mode")

class SpeechToTextService:
    """Speech-to-Text service using Vosk."""
    
    def __init__(self):
        self.model = None
        self.recognizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Vosk model."""
        try:
            logger.info(f"Loading Vosk model from {settings.VOSK_MODEL_PATH}")
            
            if VOSK_AVAILABLE:
                # For production, uncomment the following lines:
                # self.model = vosk.Model(settings.VOSK_MODEL_PATH)
                # self.recognizer = vosk.KaldiRecognizer(self.model, settings.AUDIO_SAMPLE_RATE)
                pass
            
            # Dummy model for demo
            self.model = "vosk_model_loaded"
            self.recognizer = "vosk_recognizer_ready"
            
            logger.info("Vosk STT model loaded successfully (demo mode)")
            
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            # For demo, continue with dummy model
            self.model = "dummy_model"
            self.recognizer = "dummy_recognizer"
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes (WAV format)
            
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            if not self.model or not self.recognizer:
                logger.error("STT model not loaded")
                return None
            
            # For demo purposes, return a dummy transcription
            # In production, implement actual Vosk transcription:
            """
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Process audio with Vosk
            if self.recognizer.AcceptWaveform(audio_np.tobytes()):
                result = json.loads(self.recognizer.Result())
                return result.get('text', '')
            else:
                partial = json.loads(self.recognizer.PartialResult())
                return partial.get('partial', '')
            """
            
            # Dummy transcription for demo
            dummy_transcriptions = [
                "Remind me to call mom at 3 PM",
                "Set a reminder for grocery shopping tomorrow",
                "Add a note about the meeting with John",
                "Schedule a reminder to take medication at 8 AM",
                "Create a reminder for the dentist appointment"
            ]
            
            import random
            transcription = random.choice(dummy_transcriptions)
            logger.info(f"STT transcription: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return None
    
    async def transcribe_file(self, file_path: str) -> Optional[str]:
        """
        Transcribe audio file to text.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            # For demo, just return a dummy transcription without reading the file
            return await self.transcribe_audio(b"dummy_audio_data")
            
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if the STT service is ready."""
        return self.model is not None and self.recognizer is not None 