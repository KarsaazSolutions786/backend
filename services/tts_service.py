import io
import numpy as np
from typing import Optional
from core.config import settings
from utils.logger import logger

# Try to import TFLite TTS service first
try:
    from services.tflite_tts_service import TFLiteTTSService
    TFLITE_TTS_SERVICE_AVAILABLE = True
    logger.info("TFLite TTS service available")
except ImportError:
    TFLITE_TTS_SERVICE_AVAILABLE = False
    logger.warning("TFLite TTS service not available")

# Try to import Coqui TTS first (primary engine)
try:
    from services.coqui_tts_service import CoquiTTSService
    COQUI_TTS_SERVICE_AVAILABLE = True
    logger.info("Coqui TTS service available")
except ImportError:
    COQUI_TTS_SERVICE_AVAILABLE = False
    logger.warning("Coqui TTS service not available")

# Import fallback TTS libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    logger.info("pyttsx3 library available")
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 library not available")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("gTTS library available")
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS library not available")

class TextToSpeechService:
    """
    Enhanced Text-to-Speech service with TFLite and Coqui TTS support.
    Falls back to gTTS and pyttsx3 if advanced models are not available.
    """
    
    def __init__(self):
        self.tflite_service = None
        self.coqui_service = None
        self.pyttsx3_engine = None
        self.preferred_engine = None
        self._load_engines()
    
    def _load_engines(self):
        """Load TTS engines in order of preference: TFLite -> Coqui -> gTTS -> pyttsx3."""
        try:
            # Try TFLite TTS first (uses your coqui.tflite model)
            if TFLITE_TTS_SERVICE_AVAILABLE:
                try:
                    self.tflite_service = TFLiteTTSService()
                    # Only use TFLite if it has a real model loaded (TensorFlow available + model loaded)
                    if (self.tflite_service.is_ready() and 
                        hasattr(self.tflite_service, 'is_model_loaded') and 
                        self.tflite_service.is_model_loaded):
                        self.preferred_engine = "tflite"
                        logger.info("TFLite TTS service loaded successfully as primary engine")
                        return
                    else:
                        logger.warning("TFLite TTS service has no real model loaded, trying other engines")
                        self.tflite_service = None
                except Exception as e:
                    logger.warning(f"Failed to initialize TFLite TTS: {e}")
                    self.tflite_service = None
            
            # Try Coqui TTS second (full Coqui library)
            if COQUI_TTS_SERVICE_AVAILABLE:
                try:
                    self.coqui_service = CoquiTTSService()
                    # Check if Coqui has a real model loaded (not just fallbacks)
                    if (self.coqui_service.is_ready() and 
                        hasattr(self.coqui_service, 'model') and 
                        self.coqui_service.model is not None):
                        self.preferred_engine = "coqui"
                        logger.info("Coqui TTS service loaded successfully as primary engine")
                        return
                    else:
                        logger.warning("Coqui TTS service has no real model loaded, trying fallbacks")
                        self.coqui_service = None
                except Exception as e:
                    logger.warning(f"Failed to initialize Coqui TTS: {e}")
                    self.coqui_service = None
            
            # Try gTTS as first fallback (online, but reliable)
            if GTTS_AVAILABLE:
                self.preferred_engine = "gtts"
                logger.info("Using gTTS as primary fallback TTS engine")
                return
            
            # Try pyttsx3 as final fallback (offline)
            if PYTTSX3_AVAILABLE:
                try:
                    self.pyttsx3_engine = pyttsx3.init()
                    
                    # Configure voice settings
                    voices = self.pyttsx3_engine.getProperty('voices')
                    if voices:
                        # Use first available voice
                        self.pyttsx3_engine.setProperty('voice', voices[0].id)
                    
                    # Set speech rate (words per minute)
                    self.pyttsx3_engine.setProperty('rate', 180)
                    
                    # Set volume (0.0 to 1.0)
                    self.pyttsx3_engine.setProperty('volume', 0.9)
                    
                    self.preferred_engine = "pyttsx3"
                    logger.info("pyttsx3 TTS engine loaded successfully as final fallback")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize pyttsx3: {e}")
                    self.pyttsx3_engine = None
            
            logger.error("No TTS engines available")
                
        except Exception as e:
            logger.error(f"Failed to load TTS engines: {e}")
    
    async def synthesize_speech(self, text: str, voice: str = "default") -> Optional[bytes]:
        """
        Convert text to speech audio using the best available engine.
        
        Args:
            text: Text to convert to speech
            voice: Voice model to use (optional)
            
        Returns:
            Audio data as bytes (WAV format) or None if synthesis fails
        """
        try:
            if not self.preferred_engine:
                logger.error("No TTS engine available")
                return None
            
            # Try TFLite TTS first (uses your coqui.tflite model)
            if self.preferred_engine == "tflite" and self.tflite_service:
                try:
                    return await self.tflite_service.synthesize_speech(text, voice)
                except Exception as e:
                    logger.error(f"TFLite TTS synthesis failed: {e}")
                    # Fall back to next available engine
                    return await self._fallback_synthesis(text, voice)
            
            # Try Coqui TTS second (highest quality)
            elif self.preferred_engine == "coqui" and self.coqui_service:
                try:
                    return await self.coqui_service.synthesize_speech(text, voice)
                except Exception as e:
                    logger.error(f"Coqui TTS synthesis failed: {e}")
                    # Fall back to next available engine
                    return await self._fallback_synthesis(text, voice)
            
            # Use gTTS for online TTS
            elif self.preferred_engine == "gtts" and GTTS_AVAILABLE:
                return await self._synthesize_with_gtts(text, voice)
            
            # Use pyttsx3 for offline TTS (final fallback)
            elif self.preferred_engine == "pyttsx3" and self.pyttsx3_engine:
                return await self._synthesize_with_pyttsx3(text, voice)
            
            logger.error("No working TTS engine available")
            return None
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None
    
    async def _fallback_synthesis(self, text: str, voice: str = "default") -> Optional[bytes]:
        """Try fallback synthesis methods if primary engine fails."""
        try:
            # Try gTTS fallback first
            if GTTS_AVAILABLE:
                logger.info("Falling back to gTTS for TTS synthesis")
                return await self._synthesize_with_gtts(text, voice)
            
            # Try pyttsx3 fallback
            if PYTTSX3_AVAILABLE and self.pyttsx3_engine:
                logger.info("Falling back to pyttsx3 for TTS synthesis")
                return await self._synthesize_with_pyttsx3(text, voice)
            
            logger.error("All TTS fallback methods failed")
            return None
            
        except Exception as e:
            logger.error(f"Fallback TTS synthesis failed: {e}")
            return None
    
    async def _synthesize_with_pyttsx3(self, text: str, voice: str = "default") -> Optional[bytes]:
        """Synthesize speech using pyttsx3."""
        try:
            import tempfile
            import os
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Set voice if specified and available
                if voice != "default":
                    voices = self.pyttsx3_engine.getProperty('voices')
                    for v in voices:
                        if voice.lower() in v.name.lower() or voice.lower() in v.id.lower():
                            self.pyttsx3_engine.setProperty('voice', v.id)
                            break
                
                # Save speech to file
                self.pyttsx3_engine.save_to_file(text, temp_path)
                self.pyttsx3_engine.runAndWait()
                
                # Read the generated audio file
                if os.path.exists(temp_path):
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    
                    logger.info(f"pyttsx3 TTS synthesis completed for text: '{text[:50]}...'")
                    return audio_data
                else:
                    logger.error("pyttsx3 failed to generate audio file")
                    return None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return None
    
    async def _synthesize_with_gtts(self, text: str, voice: str = "default") -> Optional[bytes]:
        """Synthesize speech using gTTS."""
        try:
            import tempfile
            import os
            
            # Map voice parameter to gTTS language codes
            lang_map = {
                "default": "en",
                "english": "en",
                "spanish": "es",
                "french": "fr",
                "german": "de",
                "italian": "it",
                "portuguese": "pt",
                "chinese": "zh",
                "japanese": "ja",
                "korean": "ko"
            }
            
            lang = lang_map.get(voice.lower(), "en")
            
            # Create gTTS object
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save TTS audio to temporary file
                tts.save(temp_path)
                
                # Read the generated audio file
                if os.path.exists(temp_path):
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    
                    logger.info(f"gTTS synthesis completed for text: '{text[:50]}...'")
                    return audio_data
                else:
                    logger.error("gTTS failed to generate audio file")
                    return None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            return None
    
    async def get_available_voices(self) -> list:
        """Get list of available voices from the current TTS engine."""
        try:
            voices = []
            
            # Get voices from TFLite TTS (primary engine)
            if self.preferred_engine == "tflite" and self.tflite_service:
                try:
                    tflite_voices = await self.tflite_service.get_available_voices()
                    if tflite_voices:
                        return tflite_voices
                except Exception as e:
                    logger.warning(f"Failed to get TFLite voices: {e}")
            
            # Get voices from Coqui TTS (secondary engine)
            if self.preferred_engine == "coqui" and self.coqui_service:
                try:
                    coqui_voices = await self.coqui_service.get_available_voices()
                    if coqui_voices:
                        return coqui_voices
                except Exception as e:
                    logger.warning(f"Failed to get Coqui voices: {e}")
            
            # Get voices from pyttsx3 (fallback)
            if self.preferred_engine == "pyttsx3" and self.pyttsx3_engine:
                pyttsx3_voices = self.pyttsx3_engine.getProperty('voices')
                for voice in pyttsx3_voices:
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "engine": "pyttsx3",
                        "lang": getattr(voice, 'languages', ['en'])
                    })
            
            # Get voices from gTTS (final fallback)
            elif self.preferred_engine == "gtts":
                voices = [
                    {"id": "en", "name": "English", "engine": "gtts", "lang": ["en"]},
                    {"id": "es", "name": "Spanish", "engine": "gtts", "lang": ["es"]},
                    {"id": "fr", "name": "French", "engine": "gtts", "lang": ["fr"]},
                    {"id": "de", "name": "German", "engine": "gtts", "lang": ["de"]},
                    {"id": "it", "name": "Italian", "engine": "gtts", "lang": ["it"]},
                    {"id": "pt", "name": "Portuguese", "engine": "gtts", "lang": ["pt"]},
                    {"id": "zh", "name": "Chinese", "engine": "gtts", "lang": ["zh"]},
                    {"id": "ja", "name": "Japanese", "engine": "gtts", "lang": ["ja"]},
                    {"id": "ko", "name": "Korean", "engine": "gtts", "lang": ["ko"]}
                ]
            
            if not voices:
                voices = [{"id": "default", "name": "Default", "engine": "fallback", "lang": ["en"]}]
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return [{"id": "default", "name": "Default", "engine": "fallback", "lang": ["en"]}]
    
    def is_ready(self) -> bool:
        """Check if the TTS service is ready."""
        return self.preferred_engine is not None
    
    def get_engine_info(self) -> dict:
        """Get information about the current TTS engine setup."""
        engine_info = {
            "primary_engine": self.preferred_engine,
            "tflite_available": TFLITE_TTS_SERVICE_AVAILABLE,
            "coqui_available": COQUI_TTS_SERVICE_AVAILABLE,
            "pyttsx3_available": PYTTSX3_AVAILABLE,
            "gtts_available": GTTS_AVAILABLE,
            "offline_capable": self.preferred_engine in ["tflite", "coqui", "pyttsx3"],
            "online_required": self.preferred_engine == "gtts"
        }
        
        # Add TFLite-specific info if available
        if self.preferred_engine == "tflite" and self.tflite_service:
            try:
                tflite_info = self.tflite_service.get_model_info()
                engine_info.update({"tflite_model_info": tflite_info})
            except Exception as e:
                logger.warning(f"Failed to get TFLite model info: {e}")
        
        # Add Coqui-specific info if available
        elif self.preferred_engine == "coqui" and self.coqui_service:
            try:
                coqui_info = self.coqui_service.get_model_info()
                engine_info.update({"coqui_model_info": coqui_info})
            except Exception as e:
                logger.warning(f"Failed to get Coqui model info: {e}")
        
        return engine_info 