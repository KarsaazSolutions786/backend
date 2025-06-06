#!/usr/bin/env python3
"""
TensorFlow Lite TTS Service
Uses the coqui.tflite model directly for text-to-speech synthesis.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from core.config import settings
from utils.logger import logger

# Try to import TensorFlow Lite
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("TensorFlow available for TFLite model loading")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - TFLite TTS will be disabled")

# Fallback imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("gTTS available as fallback")
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not available")

class TFLiteTTSService:
    """TensorFlow Lite-based TTS service for coqui.tflite model."""
    
    def __init__(self):
        self.model_path = settings.COQUI_TTS_MODEL_PATH
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_model_loaded = False
        self.available_voices = []
        self._load_model()
    
    def _load_model(self):
        """Load the TFLite model."""
        try:
            if not TF_AVAILABLE:
                logger.warning("TensorFlow not available - cannot load TFLite model")
                return False
            
            if not os.path.exists(self.model_path):
                logger.warning(f"TFLite model not found at: {self.model_path}")
                return False
            
            logger.info(f"Loading TFLite model from: {self.model_path}")
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"TFLite model loaded successfully")
            logger.info(f"Input details: {len(self.input_details)} inputs")
            logger.info(f"Output details: {len(self.output_details)} outputs")
            
            # Log model signature for debugging
            for i, input_detail in enumerate(self.input_details):
                logger.info(f"Input {i}: {input_detail['name']} - Shape: {input_detail['shape']} - Type: {input_detail['dtype']}")
            
            for i, output_detail in enumerate(self.output_details):
                logger.info(f"Output {i}: {output_detail['name']} - Shape: {output_detail['shape']} - Type: {output_detail['dtype']}")
            
            self.is_model_loaded = True
            self._setup_voices()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            self.interpreter = None
            self.is_model_loaded = False
            return False
    
    def _setup_voices(self):
        """Setup available voices for the TFLite model."""
        # For now, assume single voice model
        self.available_voices = [
            {
                "id": "tflite_default",
                "name": "Coqui TFLite Default Voice",
                "engine": "tflite",
                "lang": ["en"],
                "model": "coqui.tflite"
            }
        ]
    
    def _text_to_phonemes(self, text: str) -> np.ndarray:
        """Convert text to phoneme sequence (placeholder implementation)."""
        # This is a simplified implementation - real Coqui TTS would use
        # proper text preprocessing and phonemization
        # For now, we'll create a basic character-level encoding
        
        # Basic character to ID mapping (this would be model-specific)
        char_to_id = {
            ' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7,
            'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14,
            'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21,
            'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 27, ',': 28,
            '!': 29, '?': 30, "'": 31, '"': 32
        }
        
        # Convert text to lowercase and encode
        text = text.lower()
        phoneme_ids = []
        
        for char in text:
            if char in char_to_id:
                phoneme_ids.append(char_to_id[char])
            else:
                phoneme_ids.append(0)  # Unknown character maps to space
        
        return np.array(phoneme_ids, dtype=np.int32)
    
    async def synthesize_speech(self, text: str, voice: str = "default") -> Optional[bytes]:
        """
        Synthesize speech using the TFLite model.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (currently ignored for single-voice model)
            
        Returns:
            Audio data as bytes or None if synthesis fails
        """
        try:
            if not self.is_model_loaded:
                logger.warning("TFLite model not loaded, falling back to gTTS")
                return await self._synthesize_with_gtts(text)
            
            logger.info(f"Synthesizing speech with TFLite model: '{text[:50]}...'")
            
            # Convert text to phoneme sequence
            phoneme_ids = self._text_to_phonemes(text)
            
            # Prepare input for the model
            # Note: This is a simplified approach - real implementation would depend
            # on the exact input format expected by your specific coqui.tflite model
            
            # Reshape input to match expected model input shape
            input_shape = self.input_details[0]['shape']
            logger.info(f"Expected input shape: {input_shape}")
            
            # For demonstration, we'll create a properly shaped input
            # You may need to adjust this based on your specific model requirements
            if len(input_shape) == 2:  # [batch, sequence]
                max_length = input_shape[1] if input_shape[1] > 0 else 256
                if len(phoneme_ids) > max_length:
                    phoneme_ids = phoneme_ids[:max_length]
                else:
                    # Pad sequence to required length
                    phoneme_ids = np.pad(phoneme_ids, (0, max_length - len(phoneme_ids)), 'constant')
                
                input_data = phoneme_ids.reshape(1, -1).astype(self.input_details[0]['dtype'])
            else:
                # Handle other input shapes as needed
                input_data = phoneme_ids.reshape(input_shape).astype(self.input_details[0]['dtype'])
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            logger.info(f"TFLite inference completed. Output shape: {output_data.shape}")
            
            # Convert output to audio bytes
            # This would depend on your model's output format
            # For now, we'll create a placeholder audio response
            audio_data = self._convert_output_to_audio(output_data)
            
            if audio_data:
                logger.info(f"TFLite TTS synthesis successful - {len(audio_data)} bytes")
                return audio_data
            else:
                logger.warning("TFLite synthesis produced no audio, falling back to gTTS")
                return await self._synthesize_with_gtts(text)
                
        except Exception as e:
            logger.error(f"TFLite TTS synthesis failed: {e}")
            logger.info("Falling back to gTTS")
            return await self._synthesize_with_gtts(text)
    
    def _convert_output_to_audio(self, output_data: np.ndarray) -> Optional[bytes]:
        """Convert model output to audio bytes."""
        try:
            # This is a placeholder implementation
            # The actual conversion would depend on your model's output format
            # (e.g., mel spectrograms, raw audio, etc.)
            
            # For now, we'll return None to trigger gTTS fallback
            # Once you know the exact output format of your model,
            # this can be implemented properly
            
            logger.warning("TFLite audio conversion not implemented - using gTTS fallback")
            return None
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None
    
    async def _synthesize_with_gtts(self, text: str) -> Optional[bytes]:
        """Fallback synthesis using gTTS."""
        try:
            if not GTTS_AVAILABLE:
                logger.error("gTTS not available for fallback")
                return None
            
            import tempfile
            
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save TTS audio
                tts.save(temp_path)
                
                # Read audio data
                if os.path.exists(temp_path):
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    
                    logger.info(f"gTTS fallback synthesis completed - {len(audio_data)} bytes")
                    return audio_data
                else:
                    logger.error("gTTS failed to generate audio file")
                    return None
                    
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"gTTS fallback synthesis failed: {e}")
            return None
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        return self.available_voices
    
    def is_ready(self) -> bool:
        """Check if the TTS service is ready."""
        # Only consider ready if TensorFlow is available AND model is loaded
        # Otherwise fall back to gTTS
        if TF_AVAILABLE and self.is_model_loaded:
            return True
        elif GTTS_AVAILABLE:
            return True
        else:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the TTS model."""
        return {
            "model_type": "TensorFlow Lite",
            "model_path": self.model_path,
            "model_loaded": self.is_model_loaded,
            "tensorflow_available": TF_AVAILABLE,
            "gtts_fallback_available": GTTS_AVAILABLE,
            "input_details": [detail['name'] for detail in self.input_details] if self.input_details else [],
            "output_details": [detail['name'] for detail in self.output_details] if self.output_details else [],
            "available_voices": len(self.available_voices)
        } 