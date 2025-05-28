#!/usr/bin/env python3
"""
Test script for the Coqui STT transcription endpoint.
"""

import requests
import json
import wave
import numpy as np
import os
from pathlib import Path

def create_test_wav_file(filename: str = "test_audio.wav", duration: float = 3.0):
    """Create a simple test WAV file with the correct format."""
    
    sample_rate = 16000  # 16 kHz
    channels = 1  # Mono
    sample_width = 2  # 16-bit
    
    # Generate a simple sine wave (440 Hz - A note)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440.0  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"Created test WAV file: {filename}")
    print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz, Channels: {channels}, Bit depth: {sample_width * 8}-bit")
    
    return filename

def test_transcription_endpoint(
    base_url: str = "http://localhost:8000",
    token: str = None,
    wav_file: str = None
):
    """Test the transcription endpoint."""
    
    if not wav_file:
        # Create a test WAV file
        wav_file = create_test_wav_file()
    
    if not os.path.exists(wav_file):
        print(f"❌ WAV file not found: {wav_file}")
        return False
    
    # Prepare headers
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Test model info endpoint first
    print("\n1. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/stt/model-info", headers=headers)
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Model info retrieved successfully:")
            print(json.dumps(model_info, indent=2))
        else:
            print(f"❌ Model info failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Model info request failed: {e}")
    
    # Test transcription endpoint
    print("\n2. Testing transcription endpoint...")
    try:
        with open(wav_file, 'rb') as f:
            files = {'audio_file': (wav_file, f, 'audio/wav')}
            
            response = requests.post(
                f"{base_url}/api/v1/stt/transcribe",
                files=files,
                headers=headers
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Transcription successful:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ Transcription failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Transcription request failed: {e}")
        return False

def validate_wav_file(filename: str):
    """Validate that a WAV file meets Coqui STT requirements."""
    
    try:
        with wave.open(filename, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
            duration = frames / sample_rate
            
            print(f"\nWAV File Analysis: {filename}")
            print(f"Sample rate: {sample_rate} Hz")
            print(f"Channels: {channels}")
            print(f"Bit depth: {sample_width * 8}-bit")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Total frames: {frames}")
            
            # Check requirements
            requirements_met = True
            
            if sample_rate != 16000:
                print(f"⚠️  Sample rate should be 16000 Hz, got {sample_rate} Hz")
                requirements_met = False
            
            if channels != 1:
                print(f"⚠️  Should be mono (1 channel), got {channels} channels")
                requirements_met = False
            
            if sample_width != 2:
                print(f"⚠️  Should be 16-bit (2 bytes), got {sample_width * 8}-bit")
                requirements_met = False
            
            if requirements_met:
                print("✅ WAV file meets all Coqui STT requirements")
            else:
                print("❌ WAV file does not meet Coqui STT requirements")
            
            return requirements_met
            
    except Exception as e:
        print(f"❌ Error analyzing WAV file: {e}")
        return False

if __name__ == "__main__":
    print("Coqui STT Transcription Test")
    print("=" * 40)
    
    # You can provide your Firebase token here for authentication
    # Get this from your browser's developer tools when logged in
    firebase_token = None  # Replace with actual token if needed
    
    # Test with a custom WAV file or create one
    test_wav_file = None  # Set to path of your WAV file, or None to create a test file
    
    if test_wav_file and os.path.exists(test_wav_file):
        print(f"Using existing WAV file: {test_wav_file}")
        validate_wav_file(test_wav_file)
    else:
        print("Creating test WAV file...")
        test_wav_file = create_test_wav_file()
    
    # Run the test
    success = test_transcription_endpoint(
        base_url="http://localhost:8000",
        token=firebase_token,
        wav_file=test_wav_file
    )
    
    # Clean up test file if we created it
    if test_wav_file == "test_audio.wav":
        try:
            os.remove(test_wav_file)
            print(f"\nCleaned up test file: {test_wav_file}")
        except:
            pass
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        print("\nTroubleshooting:")
        print("1. Make sure the FastAPI server is running (python main.py)")
        print("2. Check that you have a valid Firebase token if authentication is required")
        print("3. Ensure the Coqui STT model is downloaded (python download_coqui_model.py)")
        print("4. Verify your WAV file meets the requirements (16kHz, mono, 16-bit PCM)") 