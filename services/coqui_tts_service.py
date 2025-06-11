"""
Coqui TTS Service
Integrates Coqui TTS model for high-quality text-to-speech synthesis.
"""

import os
import io
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Any, List
from pathlib import Path
from core.config import settings
from utils.logger import logger

# Import Coqui TTS
try:
    import TTS
    from TTS.api import TTS as CoquiTTS
    from TTS.utils.generic_utils import unique_filename_generator
    COQUI_TTS_AVAILABLE = True
    logger.info("Coqui TTS library available")
except ImportError:
    COQUI_TTS_AVAILABLE = False
    logger.error("Coqui TTS library not available. Install with: pip install TTS")

# Fallback imports
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    logger.info("pyttsx3 library available as fallback")
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 library not available")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("gTTS library available as fallback")
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS library not available")

class CoquiTTSService:
    """Coqui TTS-based Text-to-Speech service."""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
        self.model_path = "models/coqui.tflite"
        self.fallback_engine = None
        self.available_voices = []
        self.default_voice = "default"
        self._load_model()
    
    def _load_model(self):
        """Load the Coqui TTS model or fallback engines."""
        try:
            # Check if we have adequate dependencies for Coqui TTS
            if not COQUI_TTS_AVAILABLE:
                logger.warning("Coqui TTS not available - initializing fallback engines")
                self._initialize_fallback_engines()
                return False
            
            # Try to detect if PyTorch is available for Coqui TTS
            try:
                import torch
                pytorch_available = True
            except ImportError:
                pytorch_available = False
                logger.warning("PyTorch not available - Coqui TTS may not work properly")
            
            logger.info(f"Loading Coqui TTS model")
            
            # Check if custom model file exists
            if os.path.exists(self.model_path):
                logger.info(f"Custom Coqui model found at {self.model_path}")
                # For TFLite models, we'll use the default Coqui models for now
                # In production, you might implement custom TFLite loading
                try:
                    # Use default English TTS model
                    self.model = CoquiTTS(
                        model_name="tts_models/en/ljspeech/tacotron2-DDC",
                        progress_bar=False,
                        gpu=self.device == "cuda" and pytorch_available
                    )
                    logger.info("Coqui TTS model loaded successfully (using default English model)")
                    self._update_available_voices()
                    return True
                except Exception as e:
                    logger.error(f"Failed to load default Coqui model: {e}")
                    logger.info("Falling back to alternative TTS engines")
                    self._initialize_fallback_engines()
                    return False
            else:
                logger.info("No custom model found, trying to load default Coqui TTS model")
                if self._try_alternative_models():
                    self._update_available_voices()
                    return True
                else:
                    logger.info("All Coqui models failed, using fallback engines")
                    self._initialize_fallback_engines()
                    return False
                
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS model: {e}")
            logger.info("Initializing fallback TTS engines")
            self._initialize_fallback_engines()
            return False
    
    def _try_alternative_models(self):
        """Try to load alternative Coqui TTS models."""
        alternative_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/speedy-speech",
            "tts_models/en/ljspeech/neural_hmm",
            "tts_models/en/ljspeech/fast_pitch"
        ]
        
        for model_name in alternative_models:
            try:
                logger.info(f"Trying to load model: {model_name}")
                self.model = CoquiTTS(
                    model_name=model_name,
                    progress_bar=False,
                    gpu=self.device == "cuda"
                )
                logger.info(f"Successfully loaded Coqui TTS model: {model_name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        logger.error("Failed to load any Coqui TTS models")
        return False
    
    def _initialize_fallback_engines(self):
        """Initialize fallback TTS engines when Coqui is not available."""
        try:
            # Try pyttsx3 first
            if PYTTSX3_AVAILABLE:
                try:
                    self.fallback_engine = pyttsx3.init()
                    
                    # Configure voice settings
                    voices = self.fallback_engine.getProperty('voices')
                    if voices:
                        self.available_voices = [
                            {"id": voice.id, "name": voice.name, "lang": getattr(voice, 'languages', ['en'])}
                            for voice in voices
                        ]
                        self.fallback_engine.setProperty('voice', voices[0].id)
                    
                    # Set speech rate and volume
                    self.fallback_engine.setProperty('rate', 180)
                    self.fallback_engine.setProperty('volume', 0.9)
                    
                    logger.info("pyttsx3 fallback engine initialized")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize pyttsx3: {e}")
            
            # Try gTTS as final fallback
            if GTTS_AVAILABLE:
                self.fallback_engine = "gtts"
                self.available_voices = [
                    {"id": "gtts_en", "name": "Google TTS English", "lang": ["en"]},
                    {"id": "gtts_es", "name": "Google TTS Spanish", "lang": ["es"]},
                    {"id": "gtts_fr", "name": "Google TTS French", "lang": ["fr"]},
                ]
                logger.info("gTTS fallback engine initialized")
                return True
            
            logger.error("No TTS engines available")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback engines: {e}")
            return False
    
    def _update_available_voices(self):
        """Update available voices from Coqui TTS."""
        try:
            if self.model:
                # For single-speaker models, we have one voice
                self.available_voices = [
                    {
                        "id": "coqui_default",
                        "name": "Coqui Default Voice",
                        "lang": ["en"],
                        "model": "coqui_tts"
                    }
                ]
                
                # If multi-speaker model, get speaker list
                if hasattr(self.model, 'speakers') and self.model.speakers:
                    self.available_voices = []
                    for i, speaker in enumerate(self.model.speakers):
                        self.available_voices.append({
                            "id": f"coqui_speaker_{i}",
                            "name": f"Coqui Speaker {speaker}",
                            "lang": ["en"],
                            "model": "coqui_tts",
                            "speaker": speaker
                        })
                
                logger.info(f"Found {len(self.available_voices)} Coqui TTS voices")
                
        except Exception as e:
            logger.error(f"Failed to update available voices: {e}")
            self.available_voices = [{"id": "default", "name": "Default", "lang": ["en"]}]
    
    async def synthesize_speech(self, text: str, voice: str = "default", speed: float = 1.0) -> Optional[bytes]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID to use
            speed: Speech speed multiplier (0.5 to 2.0)
            
        Returns:
            Audio data as bytes (WAV format) or None if synthesis fails
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS")
                return None
            
            # Clean text
            text = text.strip()
            if len(text) > 1000:  # Limit text length
                text = text[:1000] + "..."
                logger.warning("Text truncated to 1000 characters")
            
            # Use Coqui TTS if available
            if self.model and COQUI_TTS_AVAILABLE:
                return await self._synthesize_with_coqui(text, voice, speed)
            
            # Use fallback engines
            elif self.fallback_engine:
                if isinstance(self.fallback_engine, str) and self.fallback_engine == "gtts":
                    return await self._synthesize_with_gtts(text, voice)
                else:
                    # Try pyttsx3 first, then fallback to gTTS if it fails
                    try:
                        audio_data = await self._synthesize_with_pyttsx3(text, voice, speed)
                        if audio_data:
                            return audio_data
                        else:
                            logger.warning("pyttsx3 failed, falling back to gTTS")
                            if GTTS_AVAILABLE:
                                return await self._synthesize_with_gtts(text, voice)
                    except Exception as e:
                        logger.warning(f"pyttsx3 failed with error: {e}, falling back to gTTS")
                        if GTTS_AVAILABLE:
                            return await self._synthesize_with_gtts(text, voice)
            
            # Final fallback: try gTTS directly if nothing else worked
            if GTTS_AVAILABLE:
                logger.info("Using gTTS as final fallback")
                return await self._synthesize_with_gtts(text, voice)
            
            logger.error("No TTS engine available")
            return None
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Even in case of error, try gTTS as last resort
            if GTTS_AVAILABLE:
                try:
                    logger.info("Attempting gTTS as last resort after error")
                    return await self._synthesize_with_gtts(text, voice)
                except Exception as gtts_error:
                    logger.error(f"Even gTTS fallback failed: {gtts_error}")
            return None
    
    async def _synthesize_with_coqui(self, text: str, voice: str = "default", speed: float = 1.0) -> Optional[bytes]:
        """Synthesize speech using Coqui TTS."""
        try:
            import tempfile
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Generate speech
                logger.info(f"Generating speech with Coqui TTS: '{text[:50]}...'")
                
                # Check if model supports multiple speakers
                if hasattr(self.model, 'speakers') and self.model.speakers and voice != "default":
                    # Find speaker by voice ID
                    speaker_name = None
                    for voice_info in self.available_voices:
                        if voice_info["id"] == voice and "speaker" in voice_info:
                            speaker_name = voice_info["speaker"]
                            break
                    
                    if speaker_name:
                        self.model.tts_to_file(
                            text=text,
                            file_path=temp_path,
                            speaker=speaker_name
                        )
                    else:
                        self.model.tts_to_file(text=text, file_path=temp_path)
                else:
                    # Single speaker model
                    self.model.tts_to_file(text=text, file_path=temp_path)
                
                # Read generated audio
                if os.path.exists(temp_path):
                    # Load audio and apply speed adjustment if needed
                    audio, sample_rate = sf.read(temp_path)
                    
                    if speed != 1.0:
                        # Simple speed adjustment by resampling
                        import librosa
                        audio = librosa.effects.time_stretch(audio, rate=speed)
                    
                    # Convert to bytes
                    audio_bytes = io.BytesIO()
                    sf.write(audio_bytes, audio, sample_rate, format='WAV')
                    audio_data = audio_bytes.getvalue()
                    
                    logger.info(f"Coqui TTS synthesis completed - {len(audio_data)} bytes")
                    return audio_data
                else:
                    logger.error("Coqui TTS failed to generate audio file")
                    return None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            return None
    
    async def _synthesize_with_pyttsx3(self, text: str, voice: str = "default", speed: float = 1.0) -> Optional[bytes]:
        """Synthesize speech using pyttsx3."""
        try:
            import tempfile
            import asyncio
            import concurrent.futures
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                def _run_pyttsx3():
                    """Run pyttsx3 in a separate thread to avoid async issues."""
                    try:
                        # Create a new engine instance for thread safety
                        import pyttsx3
                        engine = pyttsx3.init()
                        
                        # Set voice if specified
                        if voice != "default":
                            voices = engine.getProperty('voices')
                            for v in voices:
                                if voice in v.id or voice in v.name:
                                    engine.setProperty('voice', v.id)
                                    break
                        
                        # Set speed
                        rate = int(180 * speed)  # Base rate is 180 WPM
                        engine.setProperty('rate', max(50, min(400, rate)))
                        
                        # Generate speech
                        engine.save_to_file(text, temp_path)
                        engine.runAndWait()
                        
                        # Stop the engine properly
                        engine.stop()
                        del engine
                        
                        return True
                    except Exception as e:
                        logger.error(f"pyttsx3 thread execution failed: {e}")
                        return False
                
                # Run pyttsx3 in executor to avoid async loop conflicts
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    success = await loop.run_in_executor(executor, _run_pyttsx3)
                
                if not success:
                    logger.error("pyttsx3 execution failed")
                    return None
                
                # Read generated audio
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    
                    logger.info(f"pyttsx3 TTS synthesis completed - {len(audio_data)} bytes")
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
            logger.error(f"pyttsx3 TTS synthesis failed: {e}")
            return None
    
    async def _synthesize_with_gtts(self, text: str, voice: str = "default") -> Optional[bytes]:
        """Synthesize speech using Google TTS."""
        try:
            import tempfile
            
            # Map voice to language
            lang_map = {
                "gtts_en": "en",
                "gtts_es": "es", 
                "gtts_fr": "fr",
                "default": "en"
            }
            lang = lang_map.get(voice, "en")
            
            # Create gTTS object
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save TTS audio to temporary file
                tts.save(temp_path)
                
                # Read generated audio
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    
                    logger.info(f"gTTS synthesis completed - {len(audio_data)} bytes")
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
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        return self.available_voices
    
    async def generate_response_audio(self, intent: str, data: Dict[str, Any]) -> Optional[bytes]:
        """
        Generate contextual audio response based on intent and data.
        
        Args:
            intent: The classified intent
            data: Processing result data
            
        Returns:
            Audio response bytes
        """
        try:
            # Generate response text based on intent
            response_text = self._generate_response_text(intent, data)
            
            if not response_text:
                return None
            
            # Synthesize the response
            return await self.synthesize_speech(response_text, voice=self.default_voice)
            
        except Exception as e:
            logger.error(f"Response audio generation failed: {e}")
            return None
    
    def _generate_response_text(self, intent: str, data: Dict[str, Any]) -> str:
        """Generate response text based on intent and data."""
        try:
            if intent == "create_reminder":
                title = data.get('title', 'your reminder')
                time = data.get('time', '')
                person = data.get('person', '')
                
                response = f"Perfect! I've created a reminder for {title}"
                if person:
                    response += f" to contact {person}"
                if time:
                    response += f" at {time}"
                response += ". I'll make sure to notify you when the time comes!"
                return response
                
            elif intent == "create_note":
                content = data.get('content', '')
                response = "Great! I've saved your note successfully."
                if content:
                    preview = content[:30] + "..." if len(content) > 30 else content
                    response += f" Your note about '{preview}' is now safely stored."
                return response
                
            elif intent in ["create_ledger", "add_expense"]:
                amount = data.get('amount', '')
                person = data.get('person', data.get('contact_name', ''))
                
                response = "Excellent! I've recorded your financial entry"
                if amount and person:
                    response += f" for ${amount} with {person}"
                elif amount:
                    response += f" for ${amount}"
                response += ". Your records are now updated!"
                return response
                
            elif intent == "chit_chat":
                return "I understand! Is there anything specific I can help you with today? You can ask me to create reminders, notes, or track expenses."
                
            elif intent == "general_query":
                return "I've processed your request. Is there anything else you'd like me to help you with regarding reminders, notes, or your financial records?"
                
            else:
                return "I've completed your request successfully! How else can I assist you today?"
                
        except Exception as e:
            logger.error(f"Response text generation failed: {e}")
            return "Your request has been processed successfully!"
    
    def is_ready(self) -> bool:
        """Check if the TTS service is ready."""
        return (self.model is not None and COQUI_TTS_AVAILABLE) or self.fallback_engine is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "coqui_available": COQUI_TTS_AVAILABLE and self.model is not None,
            "fallback_available": self.fallback_engine is not None,
            "model_path": self.model_path,
            "device": self.device,
            "available_voices": len(self.available_voices),
            "primary_engine": "coqui_tts" if (COQUI_TTS_AVAILABLE and self.model) else "fallback",
            "fallback_engine": type(self.fallback_engine).__name__ if self.fallback_engine else None,
            "voice_list": self.available_voices
        }
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the TTS engine setup."""
        return {
            "primary_engine": "coqui_tts" if (COQUI_TTS_AVAILABLE and self.model) else "fallback",
            "coqui_available": COQUI_TTS_AVAILABLE,
            "model_loaded": self.model is not None,
            "fallback_available": self.fallback_engine is not None,
            "fallback_engine": type(self.fallback_engine).__name__ if self.fallback_engine else None,
            "pyttsx3_available": PYTTSX3_AVAILABLE,
            "gtts_available": GTTS_AVAILABLE,
            "offline_capable": self.model is not None or (hasattr(self, 'fallback_engine') and self.fallback_engine and not isinstance(self.fallback_engine, str)),
            "online_required": isinstance(self.fallback_engine, str) and self.fallback_engine == "gtts",
            "ready": self.is_ready(),
            "model_info": self.get_model_info()
        } 