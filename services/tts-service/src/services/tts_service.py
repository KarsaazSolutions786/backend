import os
import logging
import asyncio
import tempfile
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.coqui_model = None
        self.gtts_available = False
        self.pyttsx3_available = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        
        # Check library availability
        self._check_library_availability()
        
    def _check_library_availability(self):
        """Check which TTS libraries are available"""
        try:
            import gtts
            self.gtts_available = True
            logger.info("gTTS library loaded successfully")
        except ImportError:
            self.gtts_available = False
            logger.warning("gTTS library not available")

        try:
            import pyttsx3
            self.pyttsx3_available = True
            logger.info("pyttsx3 library loaded successfully")
        except ImportError:
            self.pyttsx3_available = False
            logger.warning("pyttsx3 library not available")
        
    async def load_model(self):
        """Load TTS models on startup"""
        try:
            # Use local models directory within the service
            service_models_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            coqui_model_path = os.path.join(service_models_path, "coqui.tflite")
            
            # Also check environment variable for flexibility
            env_model_path = os.getenv("MODEL_PATH")
            if env_model_path:
                alt_coqui_path = os.path.join(env_model_path, "coqui.tflite")
                if os.path.exists(alt_coqui_path):
                    coqui_model_path = alt_coqui_path

            # Try to load Coqui TTS model
            if os.path.exists(coqui_model_path):
                try:
                    # Load model in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        self._load_coqui_model,
                        coqui_model_path
                    )
                    logger.info("Coqui TTS model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Coqui model: {e}")
                    
            self.model_loaded = True
            logger.info("TTS service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            self.model_loaded = False
    
    def _load_coqui_model(self, model_path: str):
        """Load Coqui TTS model synchronously"""
        try:
            logger.info(f"Attempting to load Coqui TFLite model from: {model_path}")
            
            # Check file size to determine if it's the actual model
            file_size = os.path.getsize(model_path)
            if file_size < 1000:  # Less than 1KB, likely a Git LFS pointer
                logger.warning("Found Git LFS pointer file for Coqui model")
                logger.info("Using fallback TTS engines (gTTS/pyttsx3)")
                self.coqui_model = None
            else:
                logger.info(f"Found actual Coqui model file ({file_size / (1024*1024):.1f}MB)")
                # For actual TensorFlow Lite model, we would use TFLite interpreter
                # This is a placeholder for actual Coqui TFLite implementation
                self.coqui_model = "loaded"  # Placeholder
            
        except Exception as e:
            logger.error(f"Coqui model loading error: {e}")
            self.coqui_model = None
    
    def _synthesize_with_gtts(self, text: str, language: str) -> bytes:
        """Synthesize speech using gTTS"""
        try:
            import gtts
            tts = gtts.gTTS(text=text, lang=language, slow=False)
            
            # Save to bytes buffer
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            return fp.read()
            
        except Exception as e:
            logger.error(f"gTTS synthesis error: {e}")
            raise

    def _synthesize_with_pyttsx3(self, text: str, voice: str, speed: float) -> bytes:
        """Synthesize speech using pyttsx3"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Set properties
            engine.setProperty('rate', int(speed * 200))  # Default rate is ~200 wpm
            
            # Set voice if available
            voices = engine.getProperty('voices')
            if voices and voice != "default":
                for v in voices:
                    if voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name
            
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            # Read file content
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
            raise
    
    def _synthesize_with_coqui(self, text: str, voice: str = "default") -> bytes:
        """Synthesize speech using Coqui TTS"""
        try:
            # Placeholder for Coqui TFLite implementation
            # This would require specific TFLite inference code
            logger.info(f"Coqui TTS synthesis: {text[:50]}...")
            
            # Return placeholder for now
            return b"COQUI_AUDIO_PLACEHOLDER"
            
        except Exception as e:
            logger.error(f"Coqui synthesis error: {e}")
            raise
    
    async def synthesize(self, text: str, language: str = "en", voice: str = "default", speed: float = 1.0) -> Dict[str, Any]:
        """Synthesize speech with best available engine"""
        try:
            audio_data = None
            content_type = "audio/wav"
            engine_used = "none"
            
            # Try Coqui first if available
            if self.coqui_model:
                try:
                    loop = asyncio.get_event_loop()
                    audio_data = await loop.run_in_executor(
                        self.executor,
                        self._synthesize_with_coqui,
                        text,
                        voice
                    )
                    engine_used = "coqui"
                    content_type = "audio/wav"
                except Exception as e:
                    logger.warning(f"Coqui synthesis failed, trying fallback: {e}")
            
            # Fallback to gTTS
            if not audio_data and self.gtts_available:
                # Check if language is supported by gTTS
                gtts_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"]
                if language in gtts_languages:
                    try:
                        loop = asyncio.get_event_loop()
                        audio_data = await loop.run_in_executor(
                            self.executor,
                            self._synthesize_with_gtts,
                            text,
                            language
                        )
                        engine_used = "gtts"
                        content_type = "audio/mpeg"
                    except Exception as e:
                        logger.warning(f"gTTS synthesis failed: {e}")
            
            # Fallback to pyttsx3
            if not audio_data and self.pyttsx3_available:
                try:
                    loop = asyncio.get_event_loop()
                    audio_data = await loop.run_in_executor(
                        self.executor,
                        self._synthesize_with_pyttsx3,
                        text,
                        voice,
                        speed
                    )
                    engine_used = "pyttsx3"
                    content_type = "audio/wav"
                except Exception as e:
                    logger.warning(f"pyttsx3 synthesis failed: {e}")
            
            if not audio_data:
                raise Exception("No TTS engine available")
            
            return {
                "audio_data": audio_data,
                "content_type": content_type,
                "engine_used": engine_used,
                "text_length": len(text),
                "estimated_duration": len(text) * 0.1  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        voices = [{"id": "default", "name": "Default", "language": "en"}]
        
        if self.pyttsx3_available:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                system_voices = engine.getProperty('voices')
                for voice in system_voices:
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "language": getattr(voice, 'languages', ['en'])[0] if hasattr(voice, 'languages') else 'en'
                    })
            except:
                pass
        
        return voices
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"}
        ]
    
    def is_available(self) -> bool:
        """Check if any TTS engine is available"""
        return self.model_loaded and (self.coqui_model or self.gtts_available or self.pyttsx3_available)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all TTS engines"""
        return {
            "coqui": "available" if self.coqui_model else "unavailable",
            "gtts": "available" if self.gtts_available else "unavailable", 
            "pyttsx3": "available" if self.pyttsx3_available else "unavailable",
            "primary_engine": "coqui" if self.coqui_model else ("gtts" if self.gtts_available else "pyttsx3"),
            "model_location": "local"
        }

# Global instance
_tts_service: Optional[TTSService] = None

def get_tts_service() -> Optional[TTSService]:
    """Get the global TTS service instance"""
    return _tts_service

def initialize_tts_service() -> TTSService:
    """Initialize the global TTS service"""
    global _tts_service
    _tts_service = TTSService()
    return _tts_service 
import logging
import asyncio
import tempfile
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.coqui_model = None
        self.gtts_available = False
        self.pyttsx3_available = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_loaded = False
        
        # Check library availability
        self._check_library_availability()
        
    def _check_library_availability(self):
        """Check which TTS libraries are available"""
        try:
            import gtts
            self.gtts_available = True
            logger.info("gTTS library loaded successfully")
        except ImportError:
            self.gtts_available = False
            logger.warning("gTTS library not available")

        try:
            import pyttsx3
            self.pyttsx3_available = True
            logger.info("pyttsx3 library loaded successfully")
        except ImportError:
            self.pyttsx3_available = False
            logger.warning("pyttsx3 library not available")
        
    async def load_model(self):
        """Load TTS models on startup"""
        try:
            # Use local models directory within the service
            service_models_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            coqui_model_path = os.path.join(service_models_path, "coqui.tflite")
            
            # Also check environment variable for flexibility
            env_model_path = os.getenv("MODEL_PATH")
            if env_model_path:
                alt_coqui_path = os.path.join(env_model_path, "coqui.tflite")
                if os.path.exists(alt_coqui_path):
                    coqui_model_path = alt_coqui_path

            # Try to load Coqui TTS model
            if os.path.exists(coqui_model_path):
                try:
                    # Load model in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        self._load_coqui_model,
                        coqui_model_path
                    )
                    logger.info("Coqui TTS model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Coqui model: {e}")
                    
            self.model_loaded = True
            logger.info("TTS service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            self.model_loaded = False
    
    def _load_coqui_model(self, model_path: str):
        """Load Coqui TTS model synchronously"""
        try:
            logger.info(f"Attempting to load Coqui TFLite model from: {model_path}")
            
            # Check file size to determine if it's the actual model
            file_size = os.path.getsize(model_path)
            if file_size < 1000:  # Less than 1KB, likely a Git LFS pointer
                logger.warning("Found Git LFS pointer file for Coqui model")
                logger.info("Using fallback TTS engines (gTTS/pyttsx3)")
                self.coqui_model = None
            else:
                logger.info(f"Found actual Coqui model file ({file_size / (1024*1024):.1f}MB)")
                # For actual TensorFlow Lite model, we would use TFLite interpreter
                # This is a placeholder for actual Coqui TFLite implementation
                self.coqui_model = "loaded"  # Placeholder
            
        except Exception as e:
            logger.error(f"Coqui model loading error: {e}")
            self.coqui_model = None
    
    def _synthesize_with_gtts(self, text: str, language: str) -> bytes:
        """Synthesize speech using gTTS"""
        try:
            import gtts
            tts = gtts.gTTS(text=text, lang=language, slow=False)
            
            # Save to bytes buffer
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            return fp.read()
            
        except Exception as e:
            logger.error(f"gTTS synthesis error: {e}")
            raise

    def _synthesize_with_pyttsx3(self, text: str, voice: str, speed: float) -> bytes:
        """Synthesize speech using pyttsx3"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Set properties
            engine.setProperty('rate', int(speed * 200))  # Default rate is ~200 wpm
            
            # Set voice if available
            voices = engine.getProperty('voices')
            if voices and voice != "default":
                for v in voices:
                    if voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name
            
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            # Read file content
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
            raise
    
    def _synthesize_with_coqui(self, text: str, voice: str = "default") -> bytes:
        """Synthesize speech using Coqui TTS"""
        try:
            # Placeholder for Coqui TFLite implementation
            # This would require specific TFLite inference code
            logger.info(f"Coqui TTS synthesis: {text[:50]}...")
            
            # Return placeholder for now
            return b"COQUI_AUDIO_PLACEHOLDER"
            
        except Exception as e:
            logger.error(f"Coqui synthesis error: {e}")
            raise
    
    async def synthesize(self, text: str, language: str = "en", voice: str = "default", speed: float = 1.0) -> Dict[str, Any]:
        """Synthesize speech with best available engine"""
        try:
            audio_data = None
            content_type = "audio/wav"
            engine_used = "none"
            
            # Try Coqui first if available
            if self.coqui_model:
                try:
                    loop = asyncio.get_event_loop()
                    audio_data = await loop.run_in_executor(
                        self.executor,
                        self._synthesize_with_coqui,
                        text,
                        voice
                    )
                    engine_used = "coqui"
                    content_type = "audio/wav"
                except Exception as e:
                    logger.warning(f"Coqui synthesis failed, trying fallback: {e}")
            
            # Fallback to gTTS
            if not audio_data and self.gtts_available:
                # Check if language is supported by gTTS
                gtts_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"]
                if language in gtts_languages:
                    try:
                        loop = asyncio.get_event_loop()
                        audio_data = await loop.run_in_executor(
                            self.executor,
                            self._synthesize_with_gtts,
                            text,
                            language
                        )
                        engine_used = "gtts"
                        content_type = "audio/mpeg"
                    except Exception as e:
                        logger.warning(f"gTTS synthesis failed: {e}")
            
            # Fallback to pyttsx3
            if not audio_data and self.pyttsx3_available:
                try:
                    loop = asyncio.get_event_loop()
                    audio_data = await loop.run_in_executor(
                        self.executor,
                        self._synthesize_with_pyttsx3,
                        text,
                        voice,
                        speed
                    )
                    engine_used = "pyttsx3"
                    content_type = "audio/wav"
                except Exception as e:
                    logger.warning(f"pyttsx3 synthesis failed: {e}")
            
            if not audio_data:
                raise Exception("No TTS engine available")
            
            return {
                "audio_data": audio_data,
                "content_type": content_type,
                "engine_used": engine_used,
                "text_length": len(text),
                "estimated_duration": len(text) * 0.1  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        voices = [{"id": "default", "name": "Default", "language": "en"}]
        
        if self.pyttsx3_available:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                system_voices = engine.getProperty('voices')
                for voice in system_voices:
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "language": getattr(voice, 'languages', ['en'])[0] if hasattr(voice, 'languages') else 'en'
                    })
            except:
                pass
        
        return voices
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"}
        ]
    
    def is_available(self) -> bool:
        """Check if any TTS engine is available"""
        return self.model_loaded and (self.coqui_model or self.gtts_available or self.pyttsx3_available)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all TTS engines"""
        return {
            "coqui": "available" if self.coqui_model else "unavailable",
            "gtts": "available" if self.gtts_available else "unavailable", 
            "pyttsx3": "available" if self.pyttsx3_available else "unavailable",
            "primary_engine": "coqui" if self.coqui_model else ("gtts" if self.gtts_available else "pyttsx3"),
            "model_location": "local"
        }

# Global instance
_tts_service: Optional[TTSService] = None

def get_tts_service() -> Optional[TTSService]:
    """Get the global TTS service instance"""
    return _tts_service

def initialize_tts_service() -> TTSService:
    """Initialize the global TTS service"""
    global _tts_service
    _tts_service = TTSService()
    return _tts_service 
 
 