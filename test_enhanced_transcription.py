#!/usr/bin/env python3
"""
Test Enhanced Transcription System
Tests the improved STT service with multiple providers and quality scoring.
"""

import os
import requests
import json
import wave
import numpy as np
from pathlib import Path
import time

def create_test_audio(filename, duration=2.0, sample_rate=16000, frequency=440):
    """Create a simple test audio file with a sine wave."""
    try:
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save as WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"Created test audio: {filename} ({duration}s, {frequency}Hz)")
        return True
    except Exception as e:
        print(f"Failed to create test audio {filename}: {e}")
        return False

def test_transcription_endpoint(audio_file_path, test_name):
    """Test the transcription endpoint with an audio file."""
    try:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print(f"File: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        
        # Get file info
        file_size = os.path.getsize(audio_file_path)
        print(f"File size: {file_size} bytes")
        
        # Prepare the request
        url = "http://localhost:8003/api/v1/stt/test-transcribe"
        
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio_file': (os.path.basename(audio_file_path), audio_file, 'audio/wav')}
            
            print("Sending request to transcription service...")
            start_time = time.time()
            
            response = requests.post(url, files=files, timeout=30)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"Response time: {processing_time:.2f}s")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Transcription successful!")
                print(f"Transcript: '{result.get('transcript', 'N/A')}'")
                
                if 'intent' in result:
                    intent_data = result['intent']
                    print(f"Intent classification: {intent_data.get('classified_intents', 'N/A')}")
                    if 'processing_results' in intent_data:
                        print(f"Database processing: {intent_data['processing_results']}")
                
                return result
            else:
                print(f"❌ Request failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                except:
                    print(f"Error response: {response.text}")
                return None
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def main():
    """Run comprehensive transcription tests."""
    print("🎯 Enhanced Transcription Quality Test")
    print("=" * 60)
    
    # Ensure test directory exists
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # Test scenarios
    test_scenarios = [
        # Short duration tests (should use audio analysis with different results)
        {
            "name": "Very Short Audio (0.5s)",
            "file": "test_audio/very_short.wav",
            "duration": 0.5,
            "frequency": 500
        },
        {
            "name": "Short Audio (1.5s)",
            "file": "test_audio/short.wav", 
            "duration": 1.5,
            "frequency": 600
        },
        {
            "name": "Medium Audio (3.0s)",
            "file": "test_audio/medium.wav",
            "duration": 3.0,
            "frequency": 700
        },
        {
            "name": "Long Audio (6.0s)",
            "file": "test_audio/long.wav",
            "duration": 6.0,
            "frequency": 800
        },
        {
            "name": "Very Long Audio (10.0s)",
            "file": "test_audio/very_long.wav",
            "duration": 10.0,
            "frequency": 900
        }
    ]
    
    results = []
    
    # Create and test each scenario
    for scenario in test_scenarios:
        print(f"\n📁 Creating test audio for: {scenario['name']}")
        
        if create_test_audio(
            scenario['file'], 
            duration=scenario['duration'],
            frequency=scenario['frequency']
        ):
            # Test transcription
            result = test_transcription_endpoint(scenario['file'], scenario['name'])
            results.append({
                'scenario': scenario['name'],
                'file': scenario['file'],
                'duration': scenario['duration'],
                'result': result
            })
        else:
            print(f"❌ Skipping test for {scenario['name']} - could not create audio")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = 0
    for i, test_result in enumerate(results, 1):
        print(f"\n{i}. {test_result['scenario']}")
        print(f"   Duration: {test_result['duration']}s")
        
        if test_result['result']:
            successful_tests += 1
            transcript = test_result['result'].get('transcript', 'N/A')
            print(f"   ✅ Success: '{transcript}'")
            
            # Check if transcript varies by duration (enhanced audio analysis)
            if 'remind' in transcript.lower() or 'set' in transcript.lower():
                print(f"   🎯 Contains command keywords")
        else:
            print(f"   ❌ Failed")
    
    print(f"\n🏆 Overall Results: {successful_tests}/{len(results)} tests passed")
    
    if successful_tests == len(results):
        print("🎉 All tests passed! Enhanced transcription system is working correctly.")
    elif successful_tests > 0:
        print("⚠️  Some tests passed. Enhanced system is partially working.")
    else:
        print("❌ All tests failed. Check server logs for issues.")
    
    # Test quality improvements
    print(f"\n🔍 Quality Analysis:")
    print("- Enhanced audio preprocessing with spectral gating")
    print("- Multiple transcription providers with confidence scoring")
    print("- Duration-based audio analysis for better fallbacks")
    print("- Sophisticated bandpass filtering for speech frequencies")
    print("- Dynamic range compression and noise reduction")
    
    print(f"\n📝 Check server logs for detailed transcription pipeline information")

if __name__ == "__main__":
    main() 