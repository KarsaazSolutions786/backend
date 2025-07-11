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
        self.coqui_processor = None
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
            # Determine the model path for Coqui XTTS-v2
            MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "../../../models"))
            COQUI_MODEL_PATH = os.path.join(MODEL_PATH, "coqui_xtts_v2")

            # Try to load Coqui XTTS-v2 model
            if os.path.exists(COQUI_MODEL_PATH):
                try:
                    # Load model in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        self._load_coqui_model,
                        COQUI_MODEL_PATH
                    )
                    logger.info("Coqui XTTS-v2 model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Coqui XTTS-v2 model: {e}")
                    
            self.model_loaded = True
            logger.info("TTS service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            self.model_loaded = False
    
    def _load_coqui_model(self, model_path: str):
        """Load Coqui XTTS-v2 model synchronously"""
        try:
            logger.info(f"Loading Coqui XTTS-v2 model from: {model_path}")
            
            # Check if model files exist
            required_files = ["model.pth", "config.json", "vocab.json"]
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    logger.error(f"Required model file not found: {file_path}")
                    self.coqui_model = None
                    return
            
            # Import TTS library
            from TTS.api import TTS
            
            # Load the XTTS-v2 model from local path
            # For local models, we need to specify the model name and path
            self.coqui_model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
            logger.info(f"Coqui XTTS-v2 model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Coqui XTTS-v2 model loading error: {e}")
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
    
    def _synthesize_with_coqui(self, text: str, voice: str = "default", language: str = "en") -> bytes:
        """Synthesize speech using Coqui XTTS-v2"""
        try:
            if not self.coqui_model:
                raise Exception("Coqui XTTS-v2 model not loaded")
            
            logger.info(f"Coqui XTTS-v2 synthesis: {text[:50]}...")
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name
            
            # Synthesize speech using XTTS-v2
            # XTTS-v2 supports multilingual synthesis and voice cloning
            self.coqui_model.tts_to_file(
                text=text,
                file_path=temp_path,
                language=language,
                speaker_wav=None,  # Use default voice, can be extended for voice cloning
                voice_dir=None
            )
            
            # Read the generated audio file
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            logger.info(f"Coqui XTTS-v2 synthesis completed successfully")
            return audio_data
            
        except Exception as e:
            logger.error(f"Coqui XTTS-v2 synthesis error: {e}")
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
                        voice,
                        language
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
 
 