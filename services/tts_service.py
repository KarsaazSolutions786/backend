import io
import numpy as np
from typing import Optional
from core.config import settings
from utils.logger import logger

# Import compatible TTS libraries
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
    """Text-to-Speech service using pyttsx3 and gTTS."""
    
    def __init__(self):
        self.pyttsx3_engine = None
        self.preferred_engine = None
        self._load_engines()
    
    def _load_engines(self):
        """Load available TTS engines."""
        try:
            # Try to initialize pyttsx3 (offline TTS)
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
                    logger.info("pyttsx3 TTS engine loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize pyttsx3: {e}")
                    self.pyttsx3_engine = None
            
            # Fallback to gTTS if pyttsx3 is not available
            if not self.pyttsx3_engine and GTTS_AVAILABLE:
                self.preferred_engine = "gtts"
                logger.info("Using gTTS as TTS engine")
            
            if not self.preferred_engine:
                logger.error("No TTS engines available")
                
        except Exception as e:
            logger.error(f"Failed to load TTS engines: {e}")
    
    async def synthesize_speech(self, text: str, voice: str = "default") -> Optional[bytes]:
        """
        Convert text to speech audio.
        
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
            
            # Use pyttsx3 for offline TTS
            if self.preferred_engine == "pyttsx3" and self.pyttsx3_engine:
                return await self._synthesize_with_pyttsx3(text, voice)
            
            # Use gTTS for online TTS
            elif self.preferred_engine == "gtts" and GTTS_AVAILABLE:
                return await self._synthesize_with_gtts(text, voice)
            
            logger.error("No working TTS engine available")
            return None
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
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
        """Get list of available voices."""
        try:
            voices = []
            
            if self.preferred_engine == "pyttsx3" and self.pyttsx3_engine:
                # Get pyttsx3 voices
                pyttsx3_voices = self.pyttsx3_engine.getProperty('voices')
                for voice in pyttsx3_voices:
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "engine": "pyttsx3"
                    })
            
            elif self.preferred_engine == "gtts":
                # gTTS supported languages
                voices = [
                    {"id": "en", "name": "English", "engine": "gtts"},
                    {"id": "es", "name": "Spanish", "engine": "gtts"},
                    {"id": "fr", "name": "French", "engine": "gtts"},
                    {"id": "de", "name": "German", "engine": "gtts"},
                    {"id": "it", "name": "Italian", "engine": "gtts"},
                    {"id": "pt", "name": "Portuguese", "engine": "gtts"},
                    {"id": "zh", "name": "Chinese", "engine": "gtts"},
                    {"id": "ja", "name": "Japanese", "engine": "gtts"},
                    {"id": "ko", "name": "Korean", "engine": "gtts"}
                ]
            
            if not voices:
                voices = [{"id": "default", "name": "Default", "engine": "fallback"}]
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return [{"id": "default", "name": "Default", "engine": "fallback"}]
    
    def is_ready(self) -> bool:
        """Check if the TTS service is ready."""
        return self.preferred_engine is not None
    
    def get_engine_info(self) -> dict:
        """Get information about the TTS engine."""
        return {
            "engine": self.preferred_engine,
            "pyttsx3_available": PYTTSX3_AVAILABLE,
            "gtts_available": GTTS_AVAILABLE,
            "offline_capable": self.preferred_engine == "pyttsx3",
            "online_required": self.preferred_engine == "gtts"
        } 