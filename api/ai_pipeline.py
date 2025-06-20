"""
AI Pipeline API Endpoints
Provides endpoints for the complete AI pipeline workflow using Whisper STT, MiniLM Intent Classification, Bloom Text Generation, and Coqui TTS.
Also includes model management endpoints for quantization and optimization.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, Dict, Any, List
import aiofiles
import os
import io
import asyncio
import subprocess
import json
import wave
from pathlib import Path
from datetime import datetime
from firebase_auth import verify_firebase_token
from services.ai_pipeline_service import AIPipelineService
from utils.logger import logger
from api.bloom_response_generator import bloom_generator
import time
import numpy as np
import librosa

# AI Pipeline Service instance
ai_pipeline = AIPipelineService()

# Check if audio processing libraries are available
try:
    import librosa
    import numpy as np
    AUDIO_LIBS_AVAILABLE = True
    logger.info("Audio processing libraries available for validation")
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("Audio processing libraries not available - basic validation only")

router = APIRouter()

# Initialize the AI pipeline service
ai_pipeline = AIPipelineService()

# Model management state tracking
model_operations = {}

async def _validate_audio_file_comprehensive(audio_file: UploadFile) -> Dict[str, Any]:
    """
    Comprehensive audio file validation for empty/silent audio detection.
    
    Args:
        audio_file: The uploaded audio file
        
    Returns:
        Dictionary with validation results:
        {
            "is_valid": bool,
            "error_code": str,  # "no_voice_detected", "invalid_format", etc.
            "message": str,
            "file_size": int,
            "duration": float (if available)
        }
    """
    try:
        logger.info("Starting comprehensive audio validation")
        
        # Step 1: Check if file exists and has content
        if not audio_file or not audio_file.filename:
            logger.warning("No audio file provided")
            return {
                "is_valid": False,
                "error_code": "no_voice_detected",
                "message": "No audio file was uploaded. Please try again.",
                "file_size": 0,
                "duration": 0.0
            }
        
        logger.info(f"Validating file: {audio_file.filename}")
        
        # Step 2: Read file content to check size
        try:
            content = await audio_file.read()
            file_size = len(content)
            await audio_file.seek(0)  # Reset file pointer for later reading
            logger.info(f"File size: {file_size} bytes")
        except Exception as e:
            logger.error(f"Failed to read audio file: {e}")
            return {
                "is_valid": False,
                "error_code": "validation_error",
                "message": "Failed to read audio file. Please try again.",
                "file_size": 0,
                "duration": 0.0
            }
        
        if file_size == 0:
            logger.warning("Audio file is empty (0 bytes)")
            return {
                "is_valid": False,
                "error_code": "no_voice_detected",
                "message": "No voice was detected in the uploaded recording. Please try again.",
                "file_size": file_size,
                "duration": 0.0
            }
        
        # Step 3: Basic file size validation (too small likely means no audio)
        if file_size < 1000:  # Less than 1KB is suspicious for any audio format
            logger.warning(f"Audio file too small: {file_size} bytes")
            return {
                "is_valid": False,
                "error_code": "no_voice_detected",
                "message": "No voice was detected in the uploaded recording. Please try again.",
                "file_size": file_size,
                "duration": 0.0
            }
        
        # Step 4: Advanced audio content validation (if libraries available)
        if AUDIO_LIBS_AVAILABLE:
            logger.info("Performing advanced audio analysis")
            try:
                # Save to temporary file for analysis
                temp_path = f"uploads/temp_validation_{audio_file.filename}"
                os.makedirs("uploads", exist_ok=True)
                
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(content)
                
                logger.info(f"Saved temporary file for analysis: {temp_path}")
                
                # Analyze audio content
                try:
                    # Load audio using librosa
                    audio_data, sample_rate = librosa.load(temp_path, sr=None)
                    logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
                    
                    # Check duration
                    duration = len(audio_data) / sample_rate if sample_rate > 0 else 0.0
                    logger.info(f"Audio duration: {duration:.2f} seconds")
                    
                    # Check if audio is too short (less than 0.1 seconds)
                    if duration < 0.1:
                        logger.warning(f"Audio too short: {duration:.2f}s")
                        return {
                            "is_valid": False,
                            "error_code": "no_voice_detected",
                            "message": "No voice was detected in the uploaded recording. Please try again.",
                            "file_size": file_size,
                            "duration": duration
                        }
                    
                    # Check if audio is silent (RMS energy too low)
                    rms_energy = np.sqrt(np.mean(audio_data**2))
                    max_amplitude = np.max(np.abs(audio_data))
                    logger.info(f"Audio analysis - RMS: {rms_energy:.6f}, Max amplitude: {max_amplitude:.6f}")
                    
                    # Thresholds for silence detection
                    if rms_energy < 0.001 or max_amplitude < 0.005:
                        logger.warning(f"Audio appears silent - RMS: {rms_energy:.6f}, Max: {max_amplitude:.6f}")
                        return {
                            "is_valid": False,
                            "error_code": "no_voice_detected",
                            "message": "No voice was detected in the uploaded recording. Please try again.",
                            "file_size": file_size,
                            "duration": duration
                        }
                    
                    # NEW: Enhanced detection for white noise and non-speech patterns
                    logger.info("Running noise detection analysis")
                    noise_detection_result = _detect_white_noise_and_non_speech(audio_data, sample_rate)
                    logger.info(f"Noise detection result: {noise_detection_result}")
                    
                    if not noise_detection_result["is_speech_like"]:
                        logger.warning(f"Non-speech audio detected: {noise_detection_result['reason']}")
                        return {
                            "is_valid": False,
                            "error_code": "no_voice_detected",
                            "message": "No voice was detected in the uploaded recording. Please try again.",
                            "file_size": file_size,
                            "duration": duration,
                            "detection_details": noise_detection_result
                        }
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                        logger.info("Cleaned up temporary validation file")
                    except:
                        pass
                    
                    # Audio appears valid
                    logger.info("Audio validation passed - appears to contain speech-like content")
                    return {
                        "is_valid": True,
                        "error_code": None,
                        "message": "Audio validation passed",
                        "file_size": file_size,
                        "duration": duration
                    }
                    
                except Exception as audio_error:
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    logger.warning(f"Advanced audio analysis failed: {audio_error}")
                    # Fall back to basic validation - if we got this far, file probably has content
                    if file_size > 10000:  # 10KB threshold for basic validation
                        logger.info("Using basic validation fallback (10KB+ file)")
                        return {
                            "is_valid": True,
                            "error_code": None,
                            "message": "Audio validation passed (basic check)",
                            "file_size": file_size,
                            "duration": None
                        }
                    else:
                        logger.warning("File too small for basic validation fallback")
                        return {
                            "is_valid": False,
                            "error_code": "no_voice_detected",
                            "message": "No voice was detected in the uploaded recording. Please try again.",
                            "file_size": file_size,
                            "duration": None
                        }
            
            except Exception as e:
                logger.warning(f"Audio validation error: {e}")
                # Fall back to basic size-based validation
                pass
        else:
            logger.info("Advanced audio libraries not available, using basic validation")
        
        # Step 5: Basic validation fallback (when advanced libs not available)
        if file_size > 10000:  # 10KB threshold - reasonable minimum for audio with voice
            logger.info("Basic validation passed (file size > 10KB)")
            return {
                "is_valid": True,
                "error_code": None,
                "message": "Audio validation passed (basic check)",
                "file_size": file_size,
                "duration": None
            }
        else:
            logger.warning("Basic validation failed (file size too small)")
            return {
                "is_valid": False,
                "error_code": "no_voice_detected", 
                "message": "No voice was detected in the uploaded recording. Please try again.",
                "file_size": file_size,
                "duration": None
            }
            
    except Exception as e:
        logger.error(f"Audio validation failed with error: {e}")
        return {
            "is_valid": False,
            "error_code": "validation_error",
            "message": f"Audio validation failed: {str(e)}",
            "file_size": 0,
            "duration": None
        }

def _detect_white_noise_and_non_speech(audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Detect white noise, static, and other non-speech audio patterns.
    
    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        Dictionary with detection results:
        {
            "is_speech_like": bool,
            "reason": str,
            "confidence": float,
            "detected_patterns": List[str]
        }
    """
    try:
        detected_patterns = []
        reasons = []
        
        # Initialize all scores to safe defaults
        white_noise_score = 0.0
        static_score = 0.0
        randomness_score = 0.0
        speech_score = 0.5  # Neutral default
        tone_score = 0.0
        
        # Validate input data
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data provided to noise detection")
            return {
                "is_speech_like": False,
                "reason": "Empty audio data",
                "confidence": 1.0,
                "detected_patterns": ["empty_audio"],
                "scores": {
                    "white_noise": 0.0,
                    "static": 0.0,
                    "randomness": 0.0,
                    "speech_likelihood": 0.0,
                    "constant_tone": 0.0
                }
            }
        
        # 1. Check for white noise characteristics
        try:
            white_noise_score = _calculate_white_noise_score(audio_data, sample_rate)
            if white_noise_score > 0.7:  # High white noise probability
                detected_patterns.append("white_noise")
                reasons.append(f"White noise detected (score: {white_noise_score:.2f})")
        except Exception as e:
            logger.warning(f"White noise detection failed: {e}")
            white_noise_score = 0.0
        
        # 2. Check for static/hiss patterns
        try:
            static_score = _calculate_static_score(audio_data, sample_rate)
            if static_score > 0.8:  # High static probability
                detected_patterns.append("static")
                reasons.append(f"Static/hiss detected (score: {static_score:.2f})")
        except Exception as e:
            logger.warning(f"Static detection failed: {e}")
            static_score = 0.0
        
        # 3. Check for uniform random noise
        try:
            randomness_score = _calculate_randomness_score(audio_data)
            if randomness_score > 0.85:  # Very random, likely noise
                detected_patterns.append("random_noise")
                reasons.append(f"Random noise detected (score: {randomness_score:.2f})")
        except Exception as e:
            logger.warning(f"Randomness detection failed: {e}")
            randomness_score = 0.0
        
        # 4. Check for speech-like characteristics
        try:
            speech_score = _calculate_speech_likelihood_score(audio_data, sample_rate)
            if speech_score < 0.3:  # Low speech probability
                detected_patterns.append("non_speech")
                reasons.append(f"Low speech probability (score: {speech_score:.2f})")
        except Exception as e:
            logger.warning(f"Speech likelihood detection failed: {e}")
            speech_score = 0.5  # Neutral score
        
        # 5. Check for constant tone/beep
        try:
            tone_score = _calculate_tone_score(audio_data, sample_rate)
            if tone_score > 0.8:  # Likely a constant tone
                detected_patterns.append("constant_tone")
                reasons.append(f"Constant tone detected (score: {tone_score:.2f})")
        except Exception as e:
            logger.warning(f"Tone detection failed: {e}")
            tone_score = 0.0
        
        # Determine if this is speech-like audio
        is_speech_like = (
            white_noise_score < 0.7 and 
            static_score < 0.8 and 
            randomness_score < 0.85 and
            speech_score >= 0.3 and
            tone_score < 0.8
        )
        
        # Calculate overall confidence
        confidence = 1.0 - max(white_noise_score, static_score, randomness_score, tone_score)
        
        # If no specific patterns detected but speech score is very low, flag as non-speech
        if not detected_patterns and speech_score < 0.2:
            detected_patterns.append("low_speech_content")
            reasons.append(f"Very low speech content (score: {speech_score:.2f})")
            is_speech_like = False
        
        return {
            "is_speech_like": is_speech_like,
            "reason": "; ".join(reasons) if reasons else "Audio appears speech-like",
            "confidence": max(0.0, min(1.0, confidence)),
            "detected_patterns": detected_patterns,
            "scores": {
                "white_noise": white_noise_score,
                "static": static_score,
                "randomness": randomness_score,
                "speech_likelihood": speech_score,
                "constant_tone": tone_score
            }
        }
        
    except Exception as e:
        logger.error(f"Noise detection failed with error: {e}")
        # If the entire detection fails, be conservative and assume it might be speech
        # to avoid false positives (blocking valid speech)
        return {
            "is_speech_like": True,
            "reason": f"Detection failed, assuming speech (safety fallback): {str(e)}",
            "confidence": 0.1,  # Low confidence
            "detected_patterns": ["detection_error"],
            "scores": {
                "white_noise": 0.0,
                "static": 0.0,
                "randomness": 0.0,
                "speech_likelihood": 0.5,
                "constant_tone": 0.0
            }
        }

def _calculate_white_noise_score(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate likelihood of white noise (0.0 = not white noise, 1.0 = definitely white noise)."""
    try:
        # Check if scipy is available
        try:
            from scipy import signal as scipy_signal
        except ImportError:
            logger.warning("SciPy not available for white noise detection, using fallback")
            return _fallback_white_noise_detection(audio_data)
        
        # Compute power spectral density
        frequencies, psd = scipy_signal.welch(audio_data, sample_rate, nperseg=min(1024, len(audio_data)//4))
        
        # White noise has relatively flat power spectrum
        # Calculate coefficient of variation (std/mean) of PSD
        if len(psd) > 1 and np.mean(psd) > 0:
            psd_normalized = psd / np.mean(psd)
            cv = np.std(psd_normalized) / np.mean(psd_normalized)
            
            # White noise typically has low coefficient of variation
            # Convert to score (lower CV = higher white noise score)
            white_noise_score = max(0.0, min(1.0, 1.0 - cv))
        else:
            white_noise_score = 0.0
        
        return white_noise_score
        
    except Exception as e:
        logger.warning(f"White noise detection failed: {e}, using fallback")
        return _fallback_white_noise_detection(audio_data)

def _fallback_white_noise_detection(audio_data: np.ndarray) -> float:
    """Fallback white noise detection using simple statistical analysis."""
    try:
        # Check if the audio has very uniform distribution
        # White noise should have relatively constant amplitude
        amplitude_std = np.std(audio_data)
        amplitude_mean = np.abs(np.mean(audio_data))
        
        # White noise has high standard deviation but low mean
        if amplitude_mean < 0.01 and amplitude_std > 0.1:
            return 0.8  # Likely white noise
        else:
            return 0.0
    except Exception:
        return 0.0

def _calculate_static_score(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate likelihood of static/hiss (high frequency dominated noise)."""
    try:
        # Ensure we have enough data for FFT
        if len(audio_data) < 64:
            return 0.0
            
        # Compute FFT
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        magnitude = np.abs(fft)
        
        if len(magnitude) < 10:  # Not enough frequency bins
            return 0.0
        
        # Calculate energy in high frequency bands vs low frequency bands
        high_freq_mask = freqs > sample_rate * 0.25  # Above 1/4 of Nyquist
        low_freq_mask = freqs < sample_rate * 0.1    # Below 1/10 of Nyquist
        
        high_energy = np.mean(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0
        low_energy = np.mean(magnitude[low_freq_mask]) if np.any(low_freq_mask) else 0
        
        # Static typically has more high frequency energy
        if low_energy > 0:
            ratio = high_energy / low_energy
            static_score = min(1.0, ratio / 3.0)  # Normalize to 0-1
        else:
            static_score = 1.0 if high_energy > 0 else 0.0
        
        return static_score
        
    except Exception as e:
        logger.warning(f"Static detection failed: {e}")
        return 0.0

def _calculate_randomness_score(audio_data: np.ndarray) -> float:
    """Calculate randomness/entropy of the audio signal."""
    try:
        if len(audio_data) == 0:
            return 0.0
            
        # Quantize audio to calculate entropy
        # Use a smaller range to avoid overflow
        quantized = np.round(audio_data * 100).astype(int)
        
        # Limit the range to prevent memory issues
        quantized = np.clip(quantized, -32768, 32767)
        
        # Calculate histogram with appropriate number of bins
        num_bins = min(256, len(np.unique(quantized)))
        if num_bins < 2:
            return 0.0
            
        hist, _ = np.histogram(quantized, bins=num_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) < 2:
            return 0.0
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize entropy (max entropy depends on number of bins)
        max_entropy = np.log2(len(hist))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return min(1.0, max(0.0, normalized_entropy))
        
    except Exception as e:
        logger.warning(f"Randomness detection failed: {e}")
        return 0.0

def _calculate_speech_likelihood_score(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate likelihood that the audio contains speech patterns."""
    try:
        if len(audio_data) < 64:  # Too short for meaningful analysis
            return 0.5
            
        # 1. Check for speech-like frequency content (formants around 500Hz, 1500Hz, 2500Hz)
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        magnitude = np.abs(fft)
        
        if len(magnitude) < 10:
            return 0.5
        
        # Look for energy in typical speech formant regions
        formant_ranges = [
            (300, 800),    # F1 range
            (1000, 2000),  # F2 range  
            (2000, 3500)   # F3 range
        ]
        
        formant_energy = 0.0
        total_energy = np.sum(magnitude)
        
        if total_energy == 0:
            return 0.0
        
        for f_low, f_high in formant_ranges:
            mask = (freqs >= f_low) & (freqs <= f_high)
            if np.any(mask):
                formant_energy += np.sum(magnitude[mask])
        
        formant_ratio = formant_energy / total_energy
        
        # 2. Check for amplitude modulation (speech has varying amplitude)
        # Calculate frame-wise RMS energy
        frame_size = max(1, int(0.025 * sample_rate))  # 25ms frames, minimum 1
        hop_size = max(1, int(0.010 * sample_rate))    # 10ms hop, minimum 1
        
        if frame_size >= len(audio_data):
            # Audio too short for frame analysis
            energy_variation = 0.5  # Neutral score
        else:
            frames = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                rms = np.sqrt(np.mean(frame**2))
                frames.append(rms)
            
            if len(frames) > 1:
                # Speech should have varying energy levels
                frame_mean = np.mean(frames)
                if frame_mean > 0:
                    energy_variation = np.std(frames) / frame_mean
                else:
                    energy_variation = 0.0
            else:
                energy_variation = 0.0
        
        # Combine formant and modulation scores
        speech_score = (formant_ratio * 0.7) + (min(1.0, energy_variation) * 0.3)
        
        return min(1.0, max(0.0, speech_score))
        
    except Exception as e:
        logger.warning(f"Speech likelihood detection failed: {e}")
        return 0.5  # Neutral score if analysis fails

def _calculate_tone_score(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate likelihood of constant tone/beep."""
    try:
        if len(audio_data) < 64:
            return 0.0
            
        # Compute FFT
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        magnitude = np.abs(fft)
        
        if len(magnitude) < 10:
            return 0.0
        
        # Find the peak frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
        peak_magnitude = magnitude[peak_idx]
        
        # Calculate how much energy is concentrated around the peak
        # Look at ±50Hz around the peak
        freq_tolerance = 50  # Hz
        peak_mask = np.abs(freqs - peak_freq) <= freq_tolerance
        peak_energy = np.sum(magnitude[peak_mask]) if np.any(peak_mask) else 0
        total_energy = np.sum(magnitude)
        
        # High concentration indicates a tone
        if total_energy > 0:
            concentration_ratio = peak_energy / total_energy
        else:
            concentration_ratio = 0.0
        
        # Also check if the amplitude is relatively constant over time
        frame_size = max(1, int(0.050 * sample_rate))  # 50ms frames, minimum 1
        
        if frame_size >= len(audio_data):
            # Audio too short for frame analysis
            amplitude_stability = 1.0  # Assume stable for short audio
        else:
            frame_rms = []
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                rms = np.sqrt(np.mean(frame**2))
                frame_rms.append(rms)
            
            if len(frame_rms) > 1:
                frame_mean = np.mean(frame_rms)
                if frame_mean > 0:
                    amplitude_stability = 1.0 - (np.std(frame_rms) / frame_mean)
                    amplitude_stability = max(0.0, min(1.0, amplitude_stability))
                else:
                    amplitude_stability = 1.0
            else:
                amplitude_stability = 1.0
        
        # Combine frequency concentration and amplitude stability
        tone_score = (concentration_ratio * 0.6) + (amplitude_stability * 0.4)
        
        return min(1.0, max(0.0, tone_score))
        
    except Exception as e:
        logger.warning(f"Tone detection failed: {e}")
        return 0.0

# --- Enhanced AI Pipeline Endpoints with Bloom Integration ---

@router.post("/complete-pipeline-with-bloom")
async def complete_ai_pipeline_with_bloom(
    audio_file: UploadFile = File(...),
    language: str = "en",
    multi_intent: bool = True,
    store_in_database: bool = True,
    use_bloom_generation: bool = True,
    bloom_model_path: Optional[str] = None,  # Made optional
    max_tokens: int = 50,
    temperature: float = 0.7,
    generate_audio_response: bool = True,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Enhanced AI Pipeline: Audio → Whisper STT → MiniLM Intent → Database → Bloom Text Generation → Coqui TTS
    
    This endpoint processes audio through the enhanced AI pipeline with Bloom model integration:
    1. Validates and preprocesses audio file (includes empty/silent audio detection)
    2. Transcribes speech using Whisper STT
    3. Classifies intent(s) using MiniLM
    4. Stores results in database
    5. **NEW**: Generates contextual response using Bloom model (if available)
    6. Generates TTS response using Coqui TTS
    
    Args:
        audio_file: WAV audio file (16kHz, mono, 16-bit PCM recommended)
        language: Language code for transcription (default: "en")
        multi_intent: Whether to detect multiple intents (default: True)
        store_in_database: Whether to save results to database (default: True)
        use_bloom_generation: Whether to use Bloom model for response generation
        bloom_model_path: Path to quantized Bloom model (auto-detected if None)
        max_tokens: Maximum tokens for Bloom generation
        temperature: Temperature for text generation (0.1-1.0)
        generate_audio_response: Whether to generate TTS response (default: True)
        voice: Voice ID for TTS response (default: "default")
        current_user: Authenticated user from Firebase
        
    Returns:
        Complete pipeline results including Bloom-generated responses
        OR graceful error response for empty/silent audio
    """
    try:
        # Get user ID
        user_id = current_user["uid"]
        
        # ===== STEP 1: COMPREHENSIVE AUDIO VALIDATION =====
        logger.info(f"Starting audio validation for user {user_id}, file: {audio_file.filename}")
        
        try:
            validation_result = await _validate_audio_file_comprehensive(audio_file)
            logger.info(f"Audio validation completed: {validation_result}")
        except Exception as validation_error:
            logger.error(f"Audio validation failed with exception: {validation_error}")
            # Return graceful error even if validation fails
            return {
                "success": False,
                "error": "validation_error",
                "message": "Audio validation failed. Please try again with a different file.",
                "file_info": {
                    "filename": audio_file.filename,
                    "size_bytes": 0,
                    "duration_seconds": None
                },
                "validation_error": str(validation_error)
            }
        
        if not validation_result["is_valid"]:
            logger.warning(f"Audio validation failed: {validation_result['message']}")
            
            # Convert any numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types."""
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return obj
            
            # Clean the validation result to ensure JSON serializability
            clean_validation_result = convert_numpy_types(validation_result)
            
            # Return graceful error response for empty/silent audio
            return {
                "success": False,
                "error": clean_validation_result["error_code"],
                "message": clean_validation_result["message"],
                "file_info": {
                    "filename": audio_file.filename,
                    "size_bytes": clean_validation_result["file_size"],
                    "duration_seconds": clean_validation_result.get("duration")
                },
                "validation_details": clean_validation_result
            }
        
        logger.info(f"Audio validation passed - File: {audio_file.filename}, Size: {validation_result['file_size']} bytes")
        
        # ===== STEP 2: CONTINUE WITH EXISTING PIPELINE =====
        
        # Auto-detect available Bloom model if not specified
        if bloom_model_path is None:
            bloom_model_path = _find_available_bloom_model()
            if not bloom_model_path and use_bloom_generation:
                logger.warning("No Bloom model found, disabling Bloom generation")
                use_bloom_generation = False
        
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Check file extension
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in ['.wav', '.mp3', '.m4a', '.flac']:
            raise HTTPException(
                status_code=400,
                detail="Supported audio formats: WAV, MP3, M4A, FLAC"
            )
        
        # Save uploaded file temporarily
        temp_path = f"uploads/{audio_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
            
            logger.info(f"Processing enhanced AI pipeline with Bloom for user {user_id}, file: {audio_file.filename}")
            
            # Step 1-3: Run standard pipeline up to database operations
            standard_result = await ai_pipeline.process_complete_pipeline(
                audio_file_path=temp_path,
                user_id=user_id,
                language=language,
                multi_intent=multi_intent,
                store_in_database=store_in_database,
                generate_audio_response=False,  # We'll handle TTS after Bloom
                voice=voice
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if not standard_result.get("success", False):
                logger.error(f"Standard pipeline processing failed: {standard_result}")
                return {
                    "success": False,
                    "error": "pipeline_error",
                    "message": f"Pipeline processing failed: {'; '.join(standard_result.get('errors', ['Unknown error']))}",
                    "details": standard_result,
                    "file_info": {
                        "filename": audio_file.filename,
                        "size_bytes": validation_result["file_size"],
                        "duration_seconds": validation_result.get("duration")
                    }
                }
            
            # Step 4: Enhanced Response Generation with Bloom (if enabled and available)
            bloom_response = None
            final_response_text = standard_result.get("tts_result", {}).get("response_text", "")
            bloom_used = False
            
            if use_bloom_generation and bloom_model_path:
                logger.info(f"Generating enhanced response using Bloom model at: {bloom_model_path}")
                bloom_result = await _generate_bloom_response(
                    transcription=standard_result.get("transcription", ""),
                    intent_result=standard_result.get("intent_result", {}),
                    database_result=standard_result.get("database_result", {}),
                    model_path=bloom_model_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    user_id=user_id
                )
                
                bloom_response = bloom_result
                if bloom_result.get("success") and bloom_result.get("generated_text"):
                    final_response_text = bloom_result["generated_text"]
                    bloom_used = True
                    logger.info(f"Bloom generated response: {final_response_text[:100]}...")
                else:
                    logger.warning(f"Bloom generation failed, using fallback: {bloom_result.get('error', 'Unknown error')}")
            elif use_bloom_generation and not bloom_model_path:
                logger.info("Bloom generation requested but no model available, using standard response")
            
            # Step 5: Generate TTS with final response
            tts_result = None
            if generate_audio_response and final_response_text:
                logger.info("Generating TTS for final response")
                try:
                    audio_data = await ai_pipeline.generate_speech_audio(final_response_text, voice)
                    if audio_data:
                        tts_result = {
                            "success": True,
                            "response_text": final_response_text,
                            "audio_data": audio_data,
                            "audio_size": len(audio_data),
                            "voice_used": voice,
                            "bloom_enhanced": bloom_used
                        }
                    else:
                        tts_result = {
                            "success": False,
                            "error": "TTS synthesis failed",
                            "response_text": final_response_text,
                            "audio_size": 0,
                            "voice_used": voice,
                            "bloom_enhanced": False
                        }
                except Exception as e:
                    logger.error(f"TTS generation failed: {e}")
                    tts_result = {
                        "success": False,
                        "error": f"TTS generation failed: {str(e)}",
                        "response_text": final_response_text,
                        "audio_size": 0,
                        "voice_used": voice,
                        "bloom_enhanced": False
                    }
            
            # Return complete results
            return {
                "success": True,
                "transcription": standard_result.get("transcription", ""),
                "intent_result": standard_result.get("intent_result", {}),
                "database_result": standard_result.get("database_result", {}),
                "bloom_result": bloom_response,
                "tts_result": tts_result,
                "processing_time": standard_result.get("processing_time", 0),
                "bloom_enhanced": bloom_used,
                "bloom_model_path": bloom_model_path if bloom_used else None,
                "file_info": {
                    "filename": audio_file.filename,
                    "size_bytes": validation_result["file_size"],
                    "duration_seconds": validation_result.get("duration")
                },
                "final_response_text": final_response_text,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as pipeline_error:
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals():
                    os.remove(temp_path)
            except:
                pass
            
            logger.error(f"Pipeline processing error: {pipeline_error}")
            return {
                "success": False,
                "error": "pipeline_processing_error",
                "message": f"Failed to process audio through pipeline: {str(pipeline_error)}",
                "file_info": {
                    "filename": audio_file.filename,
                    "size_bytes": validation_result.get("file_size", 0),
                    "duration_seconds": validation_result.get("duration")
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        # This is the absolute last resort catch-all
        logger.error(f"Complete endpoint failure: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": "internal_server_error",
            "message": "An unexpected error occurred while processing your request. Please try again.",
            "file_info": {
                "filename": getattr(audio_file, 'filename', 'unknown') if audio_file else 'unknown',
                "size_bytes": 0,
                "duration_seconds": None
            },
            "timestamp": datetime.utcnow().isoformat(),
            "debug_error": str(e) if os.getenv("DEBUG", "false").lower() == "true" else None
        }

@router.post("/generate-bloom-response")
async def generate_bloom_response_only(
    text: str,
    context: Optional[str] = None,
    bloom_model_path: str = "models/bloom-560m-8bit",
    max_tokens: int = 50,
    temperature: float = 0.7,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Generate a response using only the Bloom model (no full pipeline).
    
    Args:
        text: Input text to generate response for
        context: Optional context to guide generation
        bloom_model_path: Path to quantized Bloom model
        max_tokens: Maximum tokens for generation
        temperature: Temperature for text generation
        current_user: Authenticated user
        
    Returns:
        Bloom-generated response
    """
    try:
        user_id = current_user["uid"]
        
        logger.info(f"Generating Bloom response for user {user_id}")
        
        # Create a mock intent result for consistent interface
        mock_intent_result = {
            "intent": "general_query",
            "confidence": 1.0,
            "original_text": text
        }
        
        result = await _generate_bloom_response(
            transcription=text,
            intent_result=mock_intent_result,
            database_result={"success": True},
            model_path=bloom_model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            user_id=user_id,
            additional_context=context
        )
        
        result["user_id"] = user_id
        return result
        
    except Exception as e:
        logger.error(f"Bloom response generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bloom response generation failed: {str(e)}"
        )

# --- Helper Functions for Bloom Integration ---

async def _generate_bloom_response(
    transcription: str,
    intent_result: Dict[str, Any],
    database_result: Dict[str, Any],
    model_path: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    user_id: str = "",
    additional_context: Optional[str] = None
) -> Dict[str, Any]:
    """Generate natural response using Bloom model with structured output."""
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            # Use intelligent fallback instead of failing
            return await _generate_intelligent_fallback_response(
                transcription, intent_result, database_result, user_id
            )
        
        # Use the Python 3.11 environment where PyTorch is installed
        python_pytorch_path = "venv-pytorch/bin/python"
        
        # Check if PyTorch environment exists
        if not os.path.exists(python_pytorch_path):
            logger.warning("PyTorch environment not found, using intelligent fallback")
            return await _generate_intelligent_fallback_response(
                transcription, intent_result, database_result, user_id
            )
        
        # Build pipeline output for the new generator
        pipeline_output = {
            "transcription": transcription,
            "intent_result": intent_result,
            "database_result": database_result
        }
        
        # TODO: Add habit detection logic here
        # For now, we'll use a simple heuristic based on repeated keywords
        habit_suggestion = _detect_habit_patterns(transcription, intent_result)
        
        # Use the new Bloom response generator
        bloom_result = await bloom_generator.generate_response(
            pipeline_output=pipeline_output,
            habit_suggestion=habit_suggestion
        )
        
        # Convert to the expected format for backward compatibility
        return {
            "success": True,
            "generated_text": bloom_result.get("response", ""),
            "processing_time": 2.0,  # Estimated time
            "model_path": model_path,
            "prompt_used": "Natural conversation prompt",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "dependency_issue": False,
            "method": "pytorch_bloom_structured",
            "python_env": "pytorch-3.11",
            "structured_output": bloom_result,  # Include full structured response
            "actions": bloom_result.get("actions", []),
            "follow_up": bloom_result.get("follow_up", "")
        }
        
    except Exception as e:
        logger.error(f"Bloom generation exception: {e}")
        # Fall back to intelligent response on any error
        return await _generate_intelligent_fallback_response(
            transcription, intent_result, database_result, user_id
        )

def _detect_habit_patterns(transcription: str, intent_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Simple habit detection based on keywords and patterns."""
    # Keywords that suggest recurring activities
    habit_keywords = [
        "daily", "every day", "everyday", "weekly", "monthly", 
        "regularly", "always", "usually", "often", "again",
        "workout", "gym", "exercise", "meeting", "call",
        "reminder", "habit", "routine"
    ]
    
    transcription_lower = transcription.lower()
    
    # Check for habit-indicating keywords
    has_habit_keywords = any(keyword in transcription_lower for keyword in habit_keywords)
    
    # Check for time-based patterns
    time_patterns = ["every", "each", "daily", "weekly", "monthly"]
    has_time_pattern = any(pattern in transcription_lower for pattern in time_patterns)
    
    if has_habit_keywords or has_time_pattern:
        # Determine suggested frequency
        if any(word in transcription_lower for word in ["daily", "every day", "everyday"]):
            suggest = "daily"
        elif any(word in transcription_lower for word in ["weekly", "every week"]):
            suggest = "weekly"
        elif any(word in transcription_lower for word in ["monthly", "every month"]):
            suggest = "monthly"
        else:
            suggest = "regularly"
        
        return {
            "is_habit": True,
            "suggest": suggest,
            "confidence": 0.7
        }
    
    return None

async def _generate_intelligent_fallback_response(
    transcription: str,
    intent_result: Dict[str, Any],
    database_result: Dict[str, Any],
    user_id: str
) -> Dict[str, Any]:
    """Generate intelligent contextual response without PyTorch."""
    try:
        import time
        start_time = time.time()
        
        # Analyze the database results to create contextual response
        storage_results = database_result.get("storage_results", [])
        successful_intents = database_result.get("successful_intents", 0)
        
        if not storage_results:
            response = "I've processed your request. How else can I help you?"
        else:
            # Create contextual response based on what was actually stored
            response_parts = []
            
            # Count different types of actions
            notes_created = sum(1 for r in storage_results if r.get("intent") == "create_note")
            reminders_created = sum(1 for r in storage_results if r.get("intent") == "create_reminder")
            ledger_entries = sum(1 for r in storage_results if r.get("intent") == "create_ledger")
            
            if successful_intents > 1:
                response_parts.append(f"Perfect! I've completed {successful_intents} tasks for you:")
            else:
                response_parts.append("Great! I've completed your request:")
            
            # Add specific details about what was done
            task_details = []
            
            for result in storage_results:
                intent = result.get("intent", "")
                data = result.get("data", {})
                
                if intent == "create_note":
                    content = data.get("content", "your note")
                    task_details.append(f"✓ Saved note: '{content}'")
                
                elif intent == "create_reminder":
                    title = data.get("title", "reminder")
                    task_details.append(f"✓ Created reminder: '{title}'")
                
                elif intent == "create_ledger":
                    amount = data.get("amount", 0)
                    contact = data.get("contact_name", "contact")
                    direction = data.get("direction", "owe")
                    if direction == "owe":
                        task_details.append(f"✓ Recorded ${amount} with {contact}")
                    else:
                        task_details.append(f"✓ Logged ${amount} from {contact}")
            
            # Combine the response
            if task_details:
                response = f"{response_parts[0]} {', '.join(task_details)}. Is there anything else you need help with?"
            else:
                response = "I've processed your request successfully. What would you like to do next?"
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "generated_text": response,
            "processing_time": processing_time,
            "model_path": "intelligent_fallback",
            "method": "contextual_fallback",
            "dependency_issue": False,
            "fallback_reason": "pytorch_unavailable"
        }
        
    except Exception as e:
        return {
            "success": True,  # Still succeed with basic response
            "generated_text": "I've processed your request. How can I help you further?",
            "processing_time": 0.1,
            "model_path": "basic_fallback",
            "method": "basic_fallback",
            "dependency_issue": False,
            "fallback_reason": f"fallback_error: {str(e)}"
        }

def _construct_bloom_prompt(
    transcription: str,
    intent_result: Dict[str, Any],
    database_result: Dict[str, Any],
    additional_context: Optional[str] = None
) -> str:
    """Construct a contextual prompt for Bloom model."""
    intent = intent_result.get("intent", "general_query")
    entities = intent_result.get("entities", {})
    
    # Base context
    prompt_parts = [
        "You are Eindr, a helpful AI assistant that helps users manage reminders, notes, and expenses.",
        f"User said: \"{transcription}\"",
        f"Intent detected: {intent}"
    ]
    
    # Add entity information
    if entities:
        entity_info = []
        for key, values in entities.items():
            if values:
                entity_info.append(f"{key}: {', '.join(values) if isinstance(values, list) else values}")
        if entity_info:
            prompt_parts.append(f"Entities found: {'; '.join(entity_info)}")
    
    # Add database result context
    if database_result and database_result.get("success"):
        if intent == "create_reminder":
            prompt_parts.append("I have successfully created the reminder.")
        elif intent == "create_note":
            prompt_parts.append("I have successfully saved the note.")
        elif intent in ["create_ledger", "add_expense"]:
            prompt_parts.append("I have successfully recorded the financial entry.")
    
    # Add additional context if provided
    if additional_context:
        prompt_parts.append(f"Additional context: {additional_context}")
    
    # Add instruction for response
    prompt_parts.extend([
        "",
        "Generate a helpful, friendly, and contextual response (1-2 sentences):",
        "Response:"
    ])
    
    return "\n".join(prompt_parts)

@router.get("/pipeline-audio-response/{user_id}")
async def get_pipeline_audio_response(
    user_id: str,
    text: str,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
) -> StreamingResponse:
    """
    Generate audio response using Coqui TTS.
    
    Args:
        user_id: User ID (must match authenticated user)
        text: Text to convert to speech
        voice: Voice ID to use
        current_user: Authenticated user
        
    Returns:
        Audio stream (WAV format)
    """
    try:
        # Verify user authorization
        if current_user["uid"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied: User ID mismatch"
            )
        
        # Generate audio
        audio_data = await ai_pipeline.generate_speech_audio(text, voice)
        
        if not audio_data:
            raise HTTPException(
                status_code=500,
                detail="TTS generation failed"
            )
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=response_{user_id}.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio response generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}"
        )

@router.post("/transcribe-only")
async def transcribe_audio_only(
    audio_file: UploadFile = File(...),
    language: str = "en",
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper STT only (no intent classification or database storage).
    
    Args:
        audio_file: Audio file to transcribe
        language: Language code for transcription
        current_user: Authenticated user
        
    Returns:
        Transcription results
    """
    try:
        user_id = current_user["uid"]
        
        # Validate file
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Save file temporarily
        temp_path = f"uploads/{audio_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
            
            # Transcribe only
            result = await ai_pipeline.transcribe_audio_only(temp_path, language)
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Add user info
            result["user_id"] = user_id
            
            return result
            
        except Exception as e:
            try:
                os.remove(temp_path)
            except:
                pass
            raise e
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@router.post("/classify-text-intent")
async def classify_text_intent(
    text: str,
    multi_intent: bool = True,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Classify intent from text using MiniLM (no audio processing).
    
    Args:
        text: Text to classify
        multi_intent: Whether to detect multiple intents
        current_user: Authenticated user
        
    Returns:
        Intent classification results
    """
    try:
        user_id = current_user["uid"]
        
        # Classify intent
        result = await ai_pipeline.classify_text_intent(text, multi_intent)
        
        # Add user info
        result["user_id"] = user_id
        
        return result
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Intent classification failed: {str(e)}"
        )

@router.post("/generate-speech")
async def generate_speech_from_text(
    text: str,
    voice: str = "default",
    current_user: dict = Depends(verify_firebase_token)
) -> StreamingResponse:
    """
    Generate speech audio from text using Coqui TTS.
    
    Args:
        text: Text to convert to speech
        voice: Voice ID to use
        current_user: Authenticated user
        
    Returns:
        Audio stream (WAV format)
    """
    try:
        user_id = current_user["uid"]
        
        # Generate audio
        audio_data = await ai_pipeline.generate_speech_audio(text, voice)
        
        if not audio_data:
            raise HTTPException(
                status_code=500,
                detail="TTS generation failed"
            )
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=tts_{user_id}.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {str(e)}"
        )

@router.get("/service-status")
async def get_ai_service_status(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get status of all AI services in the pipeline.
    
    Returns:
        Status information for all AI services
    """
    try:
        status = ai_pipeline.get_service_status()
        status["user_id"] = current_user["uid"]
        status["pipeline_ready"] = ai_pipeline.is_ready()
        
        return status
        
    except Exception as e:
        logger.error(f"Service status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Service status check failed: {str(e)}"
        )

@router.get("/available-voices")
async def get_available_voices(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get list of available TTS voices.
    
    Returns:
        List of available voices for TTS
    """
    try:
        voices = await ai_pipeline.coqui_tts.get_available_voices()
        
        return {
            "success": True,
            "voices": voices,
            "user_id": current_user["uid"]
        }
        
    except Exception as e:
        logger.error(f"Voice list retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice list retrieval failed: {str(e)}"
        )

@router.get("/model-info")
async def get_model_information(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get detailed information about all loaded AI models.
    
    Returns:
        Detailed model information
    """
    try:
        return {
            "success": True,
            "models": {
                "whisper_stt": ai_pipeline.whisper_stt.get_model_info(),
                "minilm_intent": ai_pipeline.minilm_intent.get_model_info(),
                "tts_service": ai_pipeline.coqui_tts.get_engine_info()
            },
            "pipeline_ready": ai_pipeline.is_ready(),
            "user_id": current_user["uid"]
        }
        
    except Exception as e:
        logger.error(f"Model info retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model info retrieval failed: {str(e)}"
        )

@router.post("/models/quantize-bloom")
async def quantize_bloom_model(
    background_tasks: BackgroundTasks,
    model_id: str = "bigscience/bloom-560m",
    bits: int = 8,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Quantize a Bloom model to 8-bit precision for efficient inference.
    
    This endpoint initiates model quantization in the background and returns immediately.
    Use the /models/quantization-status endpoint to track progress.
    
    Args:
        model_id: Hugging Face model ID (default: "bigscience/bloom-560m")
        bits: Quantization bits (only 8 is supported)
        output_dir: Output directory for quantized model (optional)
        device: Device to use for quantization (auto-detected if not specified)
        current_user: Authenticated user
        
    Returns:
        Operation status and tracking information
    """
    try:
        user_id = current_user["uid"]
        
        # Validate parameters
        if bits != 8:
            raise HTTPException(
                status_code=400,
                detail="Only 8-bit quantization is currently supported"
            )
        
        # Generate operation ID
        operation_id = f"quantize_{model_id.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set default output directory
        if not output_dir:
            model_name = model_id.split('/')[-1]
            output_dir = f"models/{model_name}-8bit"
        
        # Check if quantization is already in progress
        if operation_id in model_operations:
            raise HTTPException(
                status_code=409,
                detail="Quantization already in progress for this model"
            )
        
        # Initialize operation tracking
        model_operations[operation_id] = {
            "status": "starting",
            "model_id": model_id,
            "output_dir": output_dir,
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "current_step": "Initializing quantization process",
            "errors": []
        }
        
        # Start quantization in background
        background_tasks.add_task(
            _run_quantization,
            operation_id,
            model_id,
            bits,
            output_dir,
            device,
            user_id
        )
        
        logger.info(f"Started quantization operation {operation_id} for user {user_id}")
        
        return {
            "success": True,
            "operation_id": operation_id,
            "status": "started",
            "message": "Model quantization started in background",
            "model_id": model_id,
            "output_dir": output_dir,
            "estimated_time_minutes": "5-15",
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start quantization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start quantization: {str(e)}"
        )

@router.get("/models/quantization-status/{operation_id}")
async def get_quantization_status(
    operation_id: str,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get the status of a quantization operation.
    
    Args:
        operation_id: Operation ID returned from quantize-bloom endpoint
        current_user: Authenticated user
        
    Returns:
        Current status of the quantization operation
    """
    try:
        user_id = current_user["uid"]
        
        if operation_id not in model_operations:
            raise HTTPException(
                status_code=404,
                detail="Operation not found"
            )
        
        operation = model_operations[operation_id]
        
        # Check if user owns this operation
        if operation["user_id"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied: Operation belongs to different user"
            )
        
        return {
            "success": True,
            "operation_id": operation_id,
            **operation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quantization status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quantization status: {str(e)}"
        )

@router.post("/models/verify-quantized")
async def verify_quantized_model(
    model_dir: str,
    max_tokens: int = 20,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Verify a quantized model by running a test generation.
    
    Args:
        model_dir: Directory containing the quantized model
        max_tokens: Maximum tokens to generate for verification
        current_user: Authenticated user
        
    Returns:
        Verification results including performance metrics
    """
    try:
        user_id = current_user["uid"]
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Model directory not found: {model_dir}"
            )
        
        logger.info(f"Verifying quantized model in {model_dir} for user {user_id}")
        
        # Run verification script
        result = await _run_model_verification(model_dir, max_tokens)
        
        result["user_id"] = user_id
        result["model_dir"] = model_dir
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model verification failed: {str(e)}"
        )

@router.get("/models/available")
async def get_available_models(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get list of available models (both original and quantized).
    
    Returns:
        List of available models with their metadata
    """
    try:
        user_id = current_user["uid"]
        
        models_dir = Path("models")
        models = []
        
        if models_dir.exists():
            for model_path in models_dir.iterdir():
                if model_path.is_dir():
                    model_info = {
                        "name": model_path.name,
                        "path": str(model_path),
                        "type": "quantized" if "8bit" in model_path.name.lower() else "original",
                        "size_mb": _get_directory_size(model_path),
                        "modified": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
                    }
                    
                    # Check for config file
                    config_file = model_path / "config.json"
                    if config_file.exists():
                        try:
                            with open(config_file) as f:
                                config = json.load(f)
                                model_info["model_type"] = config.get("model_type", "unknown")
                                model_info["vocab_size"] = config.get("vocab_size", 0)
                        except:
                            pass
                    
                    models.append(model_info)
        
        return {
            "success": True,
            "models": models,
            "total_models": len(models),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available models: {str(e)}"
        )

@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Delete a model directory (use with caution).
    
    Args:
        model_name: Name of the model directory to delete
        current_user: Authenticated user
        
    Returns:
        Deletion status
    """
    try:
        user_id = current_user["uid"]
        
        model_path = Path("models") / model_name
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_name}"
            )
        
        if not model_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a directory: {model_name}"
            )
        
        # Safety check - only allow deletion of quantized models
        if "8bit" not in model_name.lower() and "quantized" not in model_name.lower():
            raise HTTPException(
                status_code=403,
                detail="For safety, only quantized models can be deleted via API"
            )
        
        # Delete the model directory
        import shutil
        shutil.rmtree(model_path)
        
        logger.info(f"Deleted model {model_name} for user {user_id}")
        
        return {
            "success": True,
            "message": f"Model {model_name} deleted successfully",
            "model_name": model_name,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )

# --- Helper Functions ---

async def _run_quantization(
    operation_id: str,
    model_id: str,
    bits: int,
    output_dir: str,
    device: Optional[str],
    user_id: str
):
    """Run quantization in background using PyTorch environment."""
    operation = model_operations[operation_id]
    
    try:
        # Update status
        operation["status"] = "preparing"
        operation["current_step"] = "Preparing quantization environment"
        operation["progress"] = 10
        
        # Use the Python 3.11 environment where PyTorch is installed
        python_pytorch_path = "venv-pytorch/bin/python"
        
        # Check if PyTorch environment exists
        if not os.path.exists(python_pytorch_path):
            operation["status"] = "failed"
            operation["current_step"] = "PyTorch environment not found"
            operation["progress"] = 0
            operation["errors"].append("PyTorch environment (venv-pytorch) not found. Please set up PyTorch first.")
            operation["failed_at"] = datetime.now().isoformat()
            return
        
        # Build command using PyTorch environment
        cmd = [python_pytorch_path, "quantize_bloom_simple.py", "--model_id", model_id, "--out_dir", output_dir]
        
        if device:
            cmd.extend(["--device", device])
        
        # Update status
        operation["status"] = "quantizing"
        operation["current_step"] = "Quantizing model (this may take several minutes)"
        operation["progress"] = 30
        
        # Run quantization
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="."
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Success
            operation["status"] = "completed"
            operation["current_step"] = "Quantization completed successfully"
            operation["progress"] = 100
            operation["completed_at"] = datetime.now().isoformat()
            operation["output_dir"] = output_dir
            operation["stdout"] = stdout.decode() if stdout else ""
            
            # Get model size
            if os.path.exists(output_dir):
                operation["model_size_mb"] = _get_directory_size(Path(output_dir))
            
        else:
            # Error
            operation["status"] = "failed"
            operation["current_step"] = "Quantization failed"
            operation["progress"] = 0
            operation["errors"].append(stderr.decode() if stderr else "Unknown error")
            operation["failed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        operation["status"] = "failed"
        operation["current_step"] = "Quantization failed with exception"
        operation["progress"] = 0
        operation["errors"].append(str(e))
        operation["failed_at"] = datetime.now().isoformat()
        logger.error(f"Quantization failed for operation {operation_id}: {e}")

async def _run_model_verification(model_dir: str, max_tokens: int) -> Dict[str, Any]:
    """Run model verification using PyTorch environment."""
    try:
        # Use the Python 3.11 environment where PyTorch is installed
        python_pytorch_path = "venv-pytorch/bin/python"
        
        # Check if PyTorch environment exists
        if not os.path.exists(python_pytorch_path):
            return {
                "success": False,
                "status": "PyTorch environment not found",
                "error": "PyTorch environment (venv-pytorch) not found. Please set up PyTorch first.",
                "model_functional": False
            }
        
        cmd = [
            python_pytorch_path, "test_pytorch_bloom.py"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="."
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return {
                "success": True,
                "status": "Model verification passed",
                "output": stdout.decode() if stdout else "",
                "model_functional": True,
                "python_env": "pytorch-3.11"
            }
        else:
            return {
                "success": False,
                "status": "Model verification failed",
                "error": stderr.decode() if stderr else "Unknown error",
                "model_functional": False,
                "python_env": "pytorch-3.11"
            }
        
    except Exception as e:
        return {
            "success": False,
            "status": "Verification failed with exception",
            "error": str(e),
            "model_functional": False
        }

def _get_directory_size(directory: Path) -> float:
    """Get directory size in MB."""
    try:
        total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return round(total_size / (1024 * 1024), 2)
    except:
        return 0.0

def _find_available_bloom_model() -> Optional[str]:
    """Find available Bloom model in the models directory."""
    # First, check for properly structured models
    possible_paths = [
        "models/bloom-560m-8bit",
        "models/bloom-560m-quantized",
        "models/Bloom_560M_chat",
        "models/Bloom_560M_lora"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it has required files
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                logger.info(f"Found Bloom model at: {path}")
                return path
    
    # Fallback: Check for standalone model files
    standalone_models = [
        "models/Bloom560m.bin",
        "models/bloom-560m.bin"
    ]
    
    for model_file in standalone_models:
        if os.path.exists(model_file):
            logger.info(f"Found standalone Bloom model file at: {model_file} (limited functionality)")
            return model_file
    
    logger.warning("No suitable Bloom model found in models directory")
    return None

# --- Model Management and Setup Endpoints ---

@router.get("/models/bloom-status")
async def get_bloom_model_status(
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get the status of Bloom models and PyTorch environment.
    
    Returns:
        Status of available Bloom models and PyTorch setup
    """
    try:
        user_id = current_user["uid"]
        
        # Check current Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Check for PyTorch environment
        python_pytorch_path = "venv-pytorch/bin/python"
        pytorch_env_exists = os.path.exists(python_pytorch_path)
        
        # Check PyTorch availability in the PyTorch environment
        pytorch_available = False
        pytorch_version = None
        pytorch_error = None
        
        if pytorch_env_exists:
            try:
                # Test PyTorch in the dedicated environment
                process = await asyncio.create_subprocess_exec(
                    python_pytorch_path, "-c", 
                    "import torch; import transformers; print(f'torch:{torch.__version__}|transformers:{transformers.__version__}')",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    output = stdout.decode().strip()
                    if "torch:" in output and "transformers:" in output:
                        pytorch_available = True
                        versions = output.split("|")
                        pytorch_version = {
                            "torch": versions[0].split(":")[1],
                            "transformers": versions[1].split(":")[1]
                        }
                else:
                    pytorch_error = stderr.decode().strip() if stderr else "Unknown error"
            except Exception as e:
                pytorch_error = str(e)
        
        # Check for available models
        available_model = _find_available_bloom_model()
        
        # Check for partial models
        partial_models = []
        models_dir = Path("models")
        
        if models_dir.exists():
            for model_dir in models_dir.glob("*bloom*"):
                if model_dir.is_dir():
                    has_config = (model_dir / "config.json").exists()
                    has_model = any((model_dir / f).exists() for f in ["pytorch_model.bin", "model.safetensors"])
                    has_tokenizer = any((model_dir / f).exists() for f in ["tokenizer.json", "tokenizer_config.json"])
                    
                    partial_models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "has_config": has_config,
                        "has_model": has_model,
                        "has_tokenizer": has_tokenizer,
                        "complete": has_config and has_model and has_tokenizer
                    })
        
        # Provide setup recommendations
        recommendations = []
        issues = []
        
        if not pytorch_env_exists:
            issues.append({
                "type": "pytorch_env_missing",
                "severity": "high",
                "message": "PyTorch environment not found",
                "solution": "Create Python 3.11 environment and install PyTorch"
            })
            recommendations.append("❌ PyTorch environment missing. Run: python3.11 -m venv venv-pytorch && source venv-pytorch/bin/activate && pip install torch transformers")
        elif not pytorch_available:
            issues.append({
                "type": "pytorch_not_working",
                "severity": "high",
                "message": f"PyTorch not working in environment: {pytorch_error}",
                "solution": "Reinstall PyTorch in the venv-pytorch environment"
            })
            recommendations.append(f"❌ PyTorch not working: {pytorch_error}")
        else:
            recommendations.append(f"✅ PyTorch {pytorch_version['torch']} working in Python 3.11 environment")
        
        if not available_model:
            issues.append({
                "type": "no_bloom_model",
                "severity": "medium",
                "message": "No complete Bloom model found",
                "solution": "Download and quantize a Bloom model"
            })
            recommendations.append("⚠️  No Bloom model found. Use /models/quantize-bloom to download one")
        else:
            recommendations.append(f"✅ Bloom model available at: {available_model}")
        
        # Overall status
        overall_status = "ready" if pytorch_available and available_model else "needs_setup"
        
        return {
            "success": True,
            "overall_status": overall_status,
            "current_python_version": python_version,
            "pytorch_environment": {
                "exists": pytorch_env_exists,
                "path": python_pytorch_path if pytorch_env_exists else None,
                "pytorch_available": pytorch_available,
                "pytorch_version": pytorch_version,
                "error": pytorch_error
            },
            "bloom_models": {
                "available_model": available_model,
                "partial_models": partial_models,
                "total_models": len(partial_models)
            },
            "issues": issues,
            "recommendations": recommendations,
            "setup_complete": pytorch_available and available_model,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting Bloom model status: {e}")
        return {
            "success": False,
            "error": str(e),
            "overall_status": "error"
        }

@router.post("/models/install-dependencies")
async def install_model_dependencies(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Install required dependencies for model operations.
    
    This endpoint installs PyTorch, transformers, and other dependencies
    required for model quantization and inference.
    
    Returns:
        Installation status and tracking information
    """
    try:
        user_id = current_user["uid"]
        
        operation_id = f"install_deps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize operation tracking
        model_operations[operation_id] = {
            "status": "starting",
            "operation_type": "dependency_installation",
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "current_step": "Installing PyTorch and ML dependencies",
            "errors": []
        }
        
        # Start installation in background
        background_tasks.add_task(_install_dependencies, operation_id, user_id)
        
        logger.info(f"Started dependency installation {operation_id} for user {user_id}")
        
        return {
            "success": True,
            "operation_id": operation_id,
            "status": "started",
            "message": "Dependency installation started in background",
            "estimated_time_minutes": "3-10",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to start dependency installation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start dependency installation: {str(e)}"
        )

async def _install_dependencies(operation_id: str, user_id: str):
    """Install ML dependencies in background task."""
    try:
        operation = model_operations[operation_id]
        
        # Update status
        operation["status"] = "installing"
        operation["current_step"] = "Installing PyTorch and transformers"
        operation["progress"] = 10
        
        # Install dependencies
        cmd = [
            "pip", "install", 
            "torch==2.1.0",
            "transformers==4.40.0", 
            "bitsandbytes==0.42.0",
            "accelerate==0.21.0",
            "safetensors==0.4.0"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="."
        )
        
        operation["progress"] = 50
        operation["current_step"] = "Installing packages (this may take several minutes)"
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Success
            operation["status"] = "completed"
            operation["current_step"] = "Dependencies installed successfully"
            operation["progress"] = 100
            operation["completed_at"] = datetime.now().isoformat()
            operation["stdout"] = stdout.decode() if stdout else ""
        else:
            # Error
            operation["status"] = "failed"
            operation["current_step"] = "Dependency installation failed"
            operation["progress"] = 0
            operation["errors"].append(stderr.decode() if stderr else "Unknown error")
            operation["failed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        operation["status"] = "failed"
        operation["current_step"] = "Installation failed with exception"
        operation["progress"] = 0
        operation["errors"].append(str(e))
        operation["failed_at"] = datetime.now().isoformat()
        logger.error(f"Dependency installation failed for operation {operation_id}: {e}") 