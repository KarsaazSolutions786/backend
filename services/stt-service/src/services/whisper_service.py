import os
import logging
import asyncio
import whisper
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WhisperService:
    def __init__(self):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        
    async def load_model(self):
        """Load Whisper model on startup"""
        try:
            # Use local models directory within the service
            service_models_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            whisper_tiny_path = os.path.join(service_models_path, "whisper-tiny.bin")
            
            # Also check environment variable for flexibility
            env_model_path = os.getenv("MODEL_PATH")
            if env_model_path:
                alt_whisper_path = os.path.join(env_model_path, "whisper-tiny.bin")
                if os.path.exists(alt_whisper_path):
                    whisper_tiny_path = alt_whisper_path
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                whisper_tiny_path
            )
            
            self.model_loaded = True
            logger.info("Whisper model loaded successfully")
            
            # Test GPU availability
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Using CPU for inference")
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None
            self.model_loaded = False
    
    def _load_model_sync(self, custom_model_path: str):
        """Synchronous model loading"""
        logger.info(f"Attempting to load Whisper model from: {custom_model_path}")
        
        if os.path.exists(custom_model_path):
            # Check if it's a Git LFS pointer file
            file_size = os.path.getsize(custom_model_path)
            if file_size < 1000:  # Less than 1KB, likely a Git LFS pointer
                with open(custom_model_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                        logger.warning("Found Git LFS pointer file for Whisper model")
                        logger.info("Loading standard Whisper tiny model...")
                        return whisper.load_model("tiny")
                    else:
                        logger.info("Found actual Whisper model file")
                        # For actual model file loading, you would implement custom loading
                        return whisper.load_model("tiny")
            else:
                logger.info(f"Found actual Whisper model file ({file_size / (1024*1024):.1f}MB)")
                # For actual custom model loading, you would implement specific loading logic
                logger.info("Loading standard Whisper tiny model...")
                return whisper.load_model("tiny")
        else:
            logger.info("Custom model not found, loading standard Whisper tiny model...")
            return whisper.load_model("tiny")
    
    def _transcribe_sync(self, file_path: str, language: str = "auto") -> Dict[str, Any]:
        """Synchronous transcription function"""
        try:
            if not self.model:
                raise Exception("Whisper model not available")
                
            # Transcribe audio
            result = self.model.transcribe(
                file_path,
                language=language if language != "auto" else None,
                task="transcribe",
                fp16=False  # Use FP32 for better compatibility
            )
            
            # Calculate average confidence from segments
            segments = result.get("segments", [])
            avg_confidence = 0.85  # Default confidence
            if segments:
                confidences = []
                for segment in segments:
                    # Estimate confidence from no_speech_prob (inverse relationship)
                    no_speech_prob = segment.get("no_speech_prob", 0.5)
                    confidence = 1.0 - no_speech_prob
                    confidences.append(confidence)
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
            
            # Calculate duration
            duration = 1.0
            if segments:
                last_segment = segments[-1]
                duration = last_segment.get("end", 1.0)
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": segments,
                "confidence": avg_confidence,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    async def transcribe(self, file_path: str, language: str = "auto") -> Dict[str, Any]:
        """Async transcription wrapper"""
        if not self.model_loaded:
            raise Exception("Whisper model not loaded")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._transcribe_sync,
            file_path,
            language
        )
        
        return result
    
    def is_available(self) -> bool:
        """Check if the model is available"""
        return self.model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "model_name": "whisper-tiny",
            "parameters": "39M",
            "languages_supported": 99,
            "max_audio_length": "30 seconds per chunk",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "status": "loaded",
            "model_location": "local"
        }

# Global instance
_whisper_service: Optional[WhisperService] = None

def get_whisper_service() -> Optional[WhisperService]:
    """Get the global Whisper service instance"""
    return _whisper_service

def initialize_whisper_service() -> WhisperService:
    """Initialize the global Whisper service"""
    global _whisper_service
    _whisper_service = WhisperService()
    return _whisper_service 
 
 