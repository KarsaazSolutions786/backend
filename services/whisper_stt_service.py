"""
Whisper Speech-to-Text Service
Integrates OpenAI Whisper model for high-quality speech transcription.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from core.config import settings
from utils.logger import logger

# Import PyTorch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch library available")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using fallback implementations")

# Import Whisper with fallback
try:
    if TORCH_AVAILABLE:
        import whisper
        WHISPER_AVAILABLE = True
        logger.info("Whisper library available")
    else:
        WHISPER_AVAILABLE = False
        logger.warning("Whisper requires PyTorch - using fallback STT")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.error("Whisper library not available. Install with: pip install openai-whisper")

# Fallback STT using speech_recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    logger.info("speech_recognition library available as fallback")
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("speech_recognition library not available")

class WhisperSTTService:
    """Whisper-based Speech-to-Text service with fallback implementations."""
    
    def __init__(self):
        self.model = None
        self.device = "cpu"  # Force CPU since PyTorch may not be available
        self.model_path = "models/whisper-tiny.bin"
        self.model_name = "tiny"  # Can be: tiny, base, small, medium, large
        self.fallback_recognizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model or fallback."""
        try:
            if WHISPER_AVAILABLE and TORCH_AVAILABLE:
                logger.info(f"Loading Whisper model on device: {self.device}")
                
                # Check if custom model file exists, otherwise use default Whisper model
                if os.path.exists(self.model_path):
                    logger.info(f"Loading custom Whisper model from {self.model_path}")
                    self.model = whisper.load_model(self.model_name, device=self.device)
                else:
                    logger.info(f"Loading default Whisper '{self.model_name}' model")
                    self.model = whisper.load_model(self.model_name, device=self.device)
                
                logger.info("Whisper model loaded successfully")
                return True
            else:
                logger.warning("Whisper/PyTorch not available - initializing fallback STT")
                return self._initialize_fallback_stt()
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.info("Falling back to alternative STT methods")
            return self._initialize_fallback_stt()
    
    def _initialize_fallback_stt(self):
        """Initialize fallback STT using speech_recognition."""
        try:
            if SPEECH_RECOGNITION_AVAILABLE:
                self.fallback_recognizer = sr.Recognizer()
                logger.info("Fallback speech recognition initialized")
                return True
            else:
                logger.error("No STT engines available")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize fallback STT: {e}")
            return False
    
    def _validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate audio file format and properties."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"Audio file not found: {file_path}"
            
            # Load audio to check properties
            audio, sr = librosa.load(file_path, sr=None)
            
            # Check duration (minimum 0.1 seconds, maximum 300 seconds)
            duration = len(audio) / sr
            if duration < 0.1:
                return False, "Audio file too short (minimum 0.1 seconds)"
            if duration > 300:
                return False, "Audio file too long (maximum 300 seconds)"
            
            # Check if audio contains actual content
            if np.max(np.abs(audio)) < 0.001:
                return False, "Audio file appears to be silent"
            
            logger.info(f"Audio validation passed - Duration: {duration:.2f}s, Sample rate: {sr}Hz")
            return True, "Valid audio file"
            
        except Exception as e:
            return False, f"Audio validation failed: {str(e)}"
    
    def _preprocess_audio(self, file_path: str) -> Optional[str]:
        """Preprocess audio for optimal Whisper transcription."""
        try:
            # Load audio with librosa (automatically converts to mono, 16kHz)
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Apply noise reduction if too quiet
            if np.max(np.abs(audio)) < 0.1:
                audio = audio * 3.0  # Boost quiet audio
                audio = np.clip(audio, -1.0, 1.0)  # Prevent clipping
            
            # Save preprocessed audio to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            sf.write(temp_path, audio, sr)
            logger.info(f"Audio preprocessed and saved to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return None
    
    async def transcribe_audio(self, file_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper or fallback STT.
        
        Args:
            file_path: Path to audio file
            language: Language code (default: "en" for English)
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Validate inputs and audio file
            is_valid, validation_message = self._validate_audio_file(file_path)
            if not is_valid:
                return {
                    "success": False,
                    "error": validation_message,
                    "transcription": "",
                    "confidence": 0.0
                }
            
            # Use Whisper if available, otherwise use fallback
            if self.model and WHISPER_AVAILABLE and TORCH_AVAILABLE:
                return await self._transcribe_with_whisper(file_path, language)
            elif self.fallback_recognizer and SPEECH_RECOGNITION_AVAILABLE:
                return await self._transcribe_with_fallback(file_path, language)
            else:
                return {
                    "success": False,
                    "error": "No STT engines available",
                    "transcription": "",
                    "confidence": 0.0
                }
                
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}",
                "transcription": "",
                "confidence": 0.0
            }
    
    async def _transcribe_with_whisper(self, file_path: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe using Whisper model."""
        import time
        start_time = time.time()
        
        try:
            # Preprocess audio
            processed_path = self._preprocess_audio(file_path)
            if not processed_path:
                return {
                    "success": False,
                    "error": "Audio preprocessing failed",
                    "transcription": "",
                    "confidence": 0.0
                }
            
            try:
                logger.info(f"Starting Whisper transcription for: {file_path}")
                
                # Transcribe with Whisper
                result = self.model.transcribe(
                    processed_path,
                    language=language,
                    task="transcribe",
                    verbose=False,
                    temperature=0.0,  # Deterministic output
                    best_of=1,
                    beam_size=1,
                    word_timestamps=True
                )
                
                # Extract transcription text
                transcription = result.get("text", "").strip()
                
                # Calculate confidence score from segments
                segments = result.get("segments", [])
                confidence = self._calculate_confidence(segments)
                
                # Clean up temporary file
                try:
                    os.unlink(processed_path)
                except:
                    pass
                
                logger.info(f"Whisper transcription completed: '{transcription[:100]}...'")
                
                return {
                    "success": True,
                    "transcription": transcription,
                    "confidence": confidence,
                    "language": result.get("language", language),
                    "segments": segments,
                    "model_used": f"whisper-{self.model_name}",
                    "device": self.device,
                    "processing_time": time.time() - start_time
                }
                
            except Exception as e:
                # Clean up temporary file
                try:
                    os.unlink(processed_path)
                except:
                    pass
                raise e
                
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                "success": False,
                "error": f"Whisper transcription failed: {str(e)}",
                "transcription": "",
                "confidence": 0.0
            }
    
    async def _transcribe_with_fallback(self, file_path: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe using fallback speech recognition."""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting fallback STT transcription for: {file_path}")
            
            # Load audio file
            with sr.AudioFile(file_path) as source:
                # Adjust for ambient noise if needed
                self.fallback_recognizer.adjust_for_ambient_noise(source, duration=0.1)
                audio = self.fallback_recognizer.record(source)
            
            # Try Google Speech Recognition first
            try:
                transcription = self.fallback_recognizer.recognize_google(audio, language=language)
                confidence = 0.8  # Default confidence for Google STT
                engine = "google"
                logger.info(f"Google STT transcription completed: '{transcription[:100]}...'")
                
            except sr.UnknownValueError:
                # Try with Sphinx (offline) if Google fails
                try:
                    transcription = self.fallback_recognizer.recognize_sphinx(audio)
                    confidence = 0.6  # Lower confidence for Sphinx
                    engine = "sphinx"
                    logger.info(f"Sphinx STT transcription completed: '{transcription[:100]}...'")
                    
                except (sr.UnknownValueError, sr.RequestError):
                    return {
                        "success": False,
                        "error": "Could not understand audio",
                        "transcription": "",
                        "confidence": 0.0
                    }
                    
            except sr.RequestError as e:
                logger.error(f"Google STT request failed: {e}")
                # Try Sphinx as backup
                try:
                    transcription = self.fallback_recognizer.recognize_sphinx(audio)
                    confidence = 0.6
                    engine = "sphinx"
                    logger.info(f"Sphinx STT fallback completed: '{transcription[:100]}...'")
                    
                except (sr.UnknownValueError, sr.RequestError):
                    return {
                        "success": False,
                        "error": f"STT service error: {str(e)}",
                        "transcription": "",
                        "confidence": 0.0
                    }
            
            return {
                "success": True,
                "transcription": transcription.strip(),
                "confidence": confidence,
                "language": language,
                "segments": [],  # Fallback doesn't provide segments
                "model_used": f"fallback-{engine}",
                "device": "cpu",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Fallback STT transcription failed: {e}")
            return {
                "success": False,
                "error": f"Fallback transcription failed: {str(e)}",
                "transcription": "",
                "confidence": 0.0
            }
    
    def _calculate_confidence(self, segments: list) -> float:
        """Calculate overall confidence score from Whisper segments."""
        try:
            if not segments:
                return 0.0
            
            # Whisper doesn't provide explicit confidence scores
            # We'll estimate based on segment characteristics
            total_confidence = 0.0
            total_duration = 0.0
            
            for segment in segments:
                # Use segment duration as weight
                duration = segment.get("end", 0) - segment.get("start", 0)
                if duration <= 0:
                    continue
                
                # Estimate confidence based on text characteristics
                text = segment.get("text", "").strip()
                segment_confidence = self._estimate_segment_confidence(text)
                
                total_confidence += segment_confidence * duration
                total_duration += duration
            
            if total_duration == 0:
                return 0.0
            
            return min(total_confidence / total_duration, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    def _estimate_segment_confidence(self, text: str) -> float:
        """Estimate confidence for a text segment."""
        try:
            if not text:
                return 0.0
            
            # Basic heuristics for confidence estimation
            confidence = 0.7  # Base confidence
            
            # Longer texts tend to be more reliable
            if len(text) > 20:
                confidence += 0.1
            elif len(text) < 5:
                confidence -= 0.2
            
            # Check for common transcription artifacts
            if any(artifact in text.lower() for artifact in ["[", "]", "inaudible", "unclear"]):
                confidence -= 0.3
            
            # Check for repeated characters (sign of poor audio)
            if any(char * 3 in text for char in "aeiou"):
                confidence -= 0.2
            
            # Check for proper sentence structure
            if text.endswith(('.', '!', '?')) and text[0].isupper():
                confidence += 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except:
            return 0.5
    
    async def transcribe_bytes(self, audio_bytes: bytes, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Audio data as bytes
            language: Language code
            
        Returns:
            Transcription results
        """
        try:
            import tempfile
            
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Transcribe the temporary file
                result = await self.transcribe_audio(temp_path, language)
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Transcription from bytes failed: {e}")
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}",
                "transcription": "",
                "confidence": 0.0
            }
    
    def is_ready(self) -> bool:
        """Check if the Whisper service is ready."""
        return self.model is not None and WHISPER_AVAILABLE
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_available": self.model is not None,
            "model_name": self.model_name,
            "device": self.device,
            "model_path": self.model_path,
            "library": "openai-whisper" if WHISPER_AVAILABLE else "not_available",
            "supported_languages": [
                "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
                "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi"
            ] if self.model else []
        } 