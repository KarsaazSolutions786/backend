"""
Empty Audio Guard Regression Test
================================

Critical regression test to ensure the AI pipeline continues to properly
handle empty/silent audio uploads after admin panel integration.

This test validates the fix that prevents 500 errors when white noise
or silent audio is uploaded, ensuring graceful JSON error responses.
"""

import pytest
import asyncio
import io
import wave
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import os

from main import app
from api.ai_pipeline import _validate_audio_file_comprehensive
from firebase_auth import verify_firebase_token

# Test client
client = TestClient(app)

class TestEmptyAudioGuard:
    """Test suite for empty audio validation regression testing."""
    
    @pytest.fixture
    def mock_firebase_user(self):
        """Mock Firebase user for authentication."""
        return {
            "uid": "test-user-123",
            "email": "test@example.com",
            "email_verified": True
        }
    
    def create_silent_audio_file(self, duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
        """Create a silent audio file for testing."""
        samples = int(duration_seconds * sample_rate)
        audio_data = np.zeros(samples, dtype=np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def create_white_noise_audio_file(self, duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
        """Create a white noise audio file for testing."""
        samples = int(duration_seconds * sample_rate)
        # Generate white noise with small amplitude to simulate low-level noise
        noise_amplitude = 0.1
        audio_data = (np.random.normal(0, noise_amplitude, samples) * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def create_empty_file(self) -> bytes:
        """Create an empty file for testing."""
        return b""
    
    @pytest.mark.asyncio
    async def test_validate_audio_silent_file(self):
        """Test that silent audio files are properly detected and rejected."""
        from fastapi import UploadFile
        
        # Create silent audio data
        silent_audio = self.create_silent_audio_file(duration_seconds=2.0)
        
        # Create UploadFile mock
        audio_file = Mock(spec=UploadFile)
        audio_file.filename = "silent_test.wav"
        audio_file.read = Mock(return_value=silent_audio)
        audio_file.seek = Mock()
        
        # Test validation
        result = await _validate_audio_file_comprehensive(audio_file)
        
        assert result is not None
        assert result["is_valid"] is False
        assert result["error_code"] == "no_voice_detected"
        assert "No voice was detected" in result["message"]
        assert result["file_size"] > 0  # Silent file still has WAV header
    
    @pytest.mark.asyncio
    async def test_validate_audio_white_noise_file(self):
        """Test that white noise audio files are properly detected and rejected."""
        from fastapi import UploadFile
        
        # Create white noise audio data
        white_noise_audio = self.create_white_noise_audio_file(duration_seconds=1.0)
        
        # Create UploadFile mock
        audio_file = Mock(spec=UploadFile)
        audio_file.filename = "white_noise_test.wav"
        audio_file.read = Mock(return_value=white_noise_audio)
        audio_file.seek = Mock()
        
        # Test validation
        result = await _validate_audio_file_comprehensive(audio_file)
        
        assert result is not None
        assert result["is_valid"] is False
        assert result["error_code"] == "no_voice_detected"
        assert "No voice was detected" in result["message"]
        assert result["file_size"] > 0
    
    @pytest.mark.asyncio
    async def test_validate_audio_empty_file(self):
        """Test that completely empty files are properly handled."""
        from fastapi import UploadFile
        
        # Create empty file
        empty_data = self.create_empty_file()
        
        # Create UploadFile mock
        audio_file = Mock(spec=UploadFile)
        audio_file.filename = "empty_test.wav"
        audio_file.read = Mock(return_value=empty_data)
        audio_file.seek = Mock()
        
        # Test validation
        result = await _validate_audio_file_comprehensive(audio_file)
        
        assert result is not None
        assert result["is_valid"] is False
        assert result["error_code"] == "no_voice_detected"
        assert "No voice was detected" in result["message"]
        assert result["file_size"] == 0
    
    @pytest.mark.asyncio
    async def test_validate_audio_no_file(self):
        """Test that missing audio files are properly handled."""
        # Test with None
        result = await _validate_audio_file_comprehensive(None)
        
        assert result is not None
        assert result["is_valid"] is False
        assert result["error_code"] == "no_voice_detected"
        assert "No audio file was uploaded" in result["message"]
    
    @patch('firebase_auth.verify_firebase_token')
    def test_ai_pipeline_endpoint_silent_audio_integration(self, mock_verify_token, mock_firebase_user):
        """
        Integration test: Ensure the complete AI pipeline endpoint returns 
        proper JSON error for silent audio (not 500 error).
        
        This is the critical regression test for the original issue.
        """
        # Mock Firebase authentication
        mock_verify_token.return_value = mock_firebase_user
        
        # Create silent audio file
        silent_audio = self.create_silent_audio_file()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(silent_audio)
            temp_file_path = temp_file.name
        
        try:
            # Test the AI pipeline endpoint
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/ai-pipeline/complete-pipeline-with-bloom",
                    files={"audio_file": ("silent_test.wav", f, "audio/wav")},
                    headers={"Authorization": "Bearer fake-token"},
                    data={
                        "language": "en",
                        "multi_intent": "true",
                        "store_in_database": "true",
                        "use_bloom_generation": "false",  # Disable to avoid model dependencies
                        "generate_audio_response": "false"
                    }
                )
            
            # Critical assertion: Should return 200 with JSON error, NOT 500
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            # Parse response
            response_data = response.json()
            
            # Validate graceful error response structure
            assert "success" in response_data
            assert response_data["success"] is False
            assert "error" in response_data
            assert response_data["error"] == "no_voice_detected"
            assert "message" in response_data
            assert "No voice was detected" in response_data["message"]
            assert "file_info" in response_data
            assert response_data["file_info"]["filename"] == "silent_test.wav"
            
            print("✅ Silent audio properly handled with graceful JSON error response")
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    @patch('firebase_auth.verify_firebase_token')
    def test_ai_pipeline_endpoint_white_noise_integration(self, mock_verify_token, mock_firebase_user):
        """
        Integration test: Ensure the complete AI pipeline endpoint returns 
        proper JSON error for white noise audio (the original issue).
        """
        # Mock Firebase authentication
        mock_verify_token.return_value = mock_firebase_user
        
        # Create white noise audio file
        white_noise_audio = self.create_white_noise_audio_file()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(white_noise_audio)
            temp_file_path = temp_file.name
        
        try:
            # Test the AI pipeline endpoint
            with open(temp_file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/ai-pipeline/complete-pipeline-with-bloom",
                    files={"audio_file": ("white_noise_test.wav", f, "audio/wav")},
                    headers={"Authorization": "Bearer fake-token"},
                    data={
                        "language": "en",
                        "multi_intent": "true",
                        "store_in_database": "true",
                        "use_bloom_generation": "false",  # Disable to avoid model dependencies
                        "generate_audio_response": "false"
                    }
                )
            
            # Critical assertion: Should return 200 with JSON error, NOT 500
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            # Parse response
            response_data = response.json()
            
            # Validate graceful error response structure
            assert "success" in response_data
            assert response_data["success"] is False
            assert "error" in response_data
            assert response_data["error"] == "no_voice_detected"
            assert "message" in response_data
            assert "No voice was detected" in response_data["message"]
            
            # Should include detection details for white noise
            if "validation_details" in response_data:
                validation_details = response_data["validation_details"]
                if "detection_details" in validation_details:
                    detection_details = validation_details["detection_details"]
                    assert "is_speech_like" in detection_details
                    assert detection_details["is_speech_like"] is False
                    print(f"✅ White noise detection details: {detection_details.get('reason', 'No reason provided')}")
            
            print("✅ White noise properly handled with graceful JSON error response")
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    @patch('firebase_auth.verify_firebase_token')
    def test_ai_pipeline_endpoint_empty_file_integration(self, mock_verify_token, mock_firebase_user):
        """
        Integration test: Ensure empty files are handled gracefully.
        """
        # Mock Firebase authentication
        mock_verify_token.return_value = mock_firebase_user
        
        # Create empty file
        empty_data = self.create_empty_file()
        
        # Test the AI pipeline endpoint
        response = client.post(
            "/api/v1/ai-pipeline/complete-pipeline-with-bloom",
            files={"audio_file": ("empty_test.wav", io.BytesIO(empty_data), "audio/wav")},
            headers={"Authorization": "Bearer fake-token"},
            data={
                "language": "en",
                "multi_intent": "true",
                "store_in_database": "true",
                "use_bloom_generation": "false",
                "generate_audio_response": "false"
            }
        )
        
        # Should return 200 with JSON error, NOT 500
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        # Parse response
        response_data = response.json()
        
        # Validate graceful error response structure
        assert response_data["success"] is False
        assert response_data["error"] == "no_voice_detected"
        assert "No voice was detected" in response_data["message"]
        assert response_data["file_info"]["size_bytes"] == 0
        
        print("✅ Empty file properly handled with graceful JSON error response")
    
    def test_audio_validation_performance(self):
        """Test that audio validation completes within reasonable time."""
        import time
        from fastapi import UploadFile
        
        # Create test audio file
        test_audio = self.create_white_noise_audio_file(duration_seconds=5.0)
        
        # Create UploadFile mock
        audio_file = Mock(spec=UploadFile)
        audio_file.filename = "performance_test.wav"
        audio_file.read = Mock(return_value=test_audio)
        audio_file.seek = Mock()
        
        # Measure validation time
        start_time = time.time()
        
        # Run validation in asyncio
        async def run_validation():
            return await _validate_audio_file_comprehensive(audio_file)
        
        result = asyncio.run(run_validation())
        
        end_time = time.time()
        validation_time = end_time - start_time
        
        # Should complete within 2 seconds for a 5-second audio file
        assert validation_time < 2.0, f"Validation took {validation_time:.2f}s, expected < 2.0s"
        assert result is not None
        
        print(f"✅ Audio validation completed in {validation_time:.3f}s")
    
    def test_numpy_serialization_fix(self):
        """
        Test that numpy types are properly converted for JSON serialization.
        This tests the fix for the PydanticSerializationError.
        """
        from api.ai_pipeline import convert_numpy_types
        
        # Test data with numpy types
        test_data = {
            "numpy_bool": np.bool_(True),
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(3.14),
            "numpy_array": np.array([1, 2, 3]),
            "nested": {
                "numpy_bool": np.bool_(False),
                "regular_data": "test"
            },
            "list_with_numpy": [np.int32(1), np.float32(2.5), "string"],
            "regular_data": "should_remain_unchanged"
        }
        
        # Convert numpy types
        converted = convert_numpy_types(test_data)
        
        # Validate conversions
        assert isinstance(converted["numpy_bool"], bool)
        assert converted["numpy_bool"] is True
        
        assert isinstance(converted["numpy_int"], int)
        assert converted["numpy_int"] == 42
        
        assert isinstance(converted["numpy_float"], float)
        assert abs(converted["numpy_float"] - 3.14) < 0.001
        
        assert isinstance(converted["numpy_array"], list)
        assert converted["numpy_array"] == [1, 2, 3]
        
        assert isinstance(converted["nested"]["numpy_bool"], bool)
        assert converted["nested"]["numpy_bool"] is False
        assert converted["nested"]["regular_data"] == "test"
        
        assert isinstance(converted["list_with_numpy"][0], int)
        assert isinstance(converted["list_with_numpy"][1], float)
        assert converted["list_with_numpy"][2] == "string"
        
        assert converted["regular_data"] == "should_remain_unchanged"
        
        # Ensure result is JSON serializable
        import json
        json_str = json.dumps(converted)
        assert json_str is not None
        
        print("✅ Numpy types properly converted for JSON serialization")

# Helper function to run specific tests
def run_regression_tests():
    """Run critical regression tests for empty audio handling."""
    import subprocess
    import sys
    
    print("🧪 Running Empty Audio Guard Regression Tests...")
    
    # Run pytest with specific markers
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/admin/test_empty_audio_guard.py",
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All regression tests passed!")
        print(result.stdout)
    else:
        print("❌ Some regression tests failed:")
        print(result.stdout)
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    # Run regression tests directly
    success = run_regression_tests()
    exit(0 if success else 1) 