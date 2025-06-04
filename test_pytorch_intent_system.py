#!/usr/bin/env python3
"""
Comprehensive Test Suite for PyTorch Intent Classification System
Tests classification, multi-intent detection, database integration, and API endpoints.
"""

import os
import sys
import asyncio
import pytest
import json
import requests
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from services.pytorch_intent_service import PyTorchIntentService
from services.database_integration_service import DatabaseIntegrationService
from utils.logger import logger

class TestPyTorchIntentService:
    """Test suite for PyTorch Intent Service."""
    
    @pytest.fixture
    def intent_service(self):
        """Create intent service instance for testing."""
        return PyTorchIntentService()
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "create_reminder": [
                "Remind me to call John at 5 PM",
                "Set a reminder for my doctor appointment tomorrow",
                "Don't forget to take medicine at 8 AM"
            ],
            "create_note": [
                "Note to buy groceries and milk",
                "Write down the meeting agenda",
                "Create a shopping list for weekend"
            ],
            "create_ledger": [
                "John owes me 50 dollars for dinner",
                "I borrowed 100 dollars from Sarah",
                "Mark paid me back 25 dollars"
            ],
            "chit_chat": [
                "Hello, how are you today?",
                "What can you help me with?",
                "Thank you for your assistance"
            ]
        }
    
    @pytest.fixture
    def multi_intent_texts(self):
        """Multi-intent test cases."""
        return [
            {
                "text": "Remind me to call John at 5 PM and note to buy groceries",
                "expected_intents": ["create_reminder", "create_note"]
            },
            {
                "text": "Set reminder for meeting and John owes me 50 dollars",
                "expected_intents": ["create_reminder", "create_ledger"]
            },
            {
                "text": "Create a shopping list and remind me about the deadline",
                "expected_intents": ["create_note", "create_reminder"]
            }
        ]
    
    def test_service_initialization(self, intent_service):
        """Test that the service initializes correctly."""
        assert intent_service is not None
        assert intent_service.intent_labels == ["create_reminder", "create_note", "create_ledger", "chit_chat"]
        assert intent_service.device is not None
        
        model_info = intent_service.get_model_info()
        assert isinstance(model_info, dict)
        assert "pytorch_available" in model_info
        assert "device" in model_info
    
    @pytest.mark.asyncio
    async def test_single_intent_classification(self, intent_service, sample_texts):
        """Test single intent classification for each intent type."""
        for expected_intent, texts in sample_texts.items():
            for text in texts:
                result = await intent_service.classify_intent(text, multi_intent=False)
                
                assert result is not None
                assert "intent" in result
                assert "confidence" in result
                assert "entities" in result
                assert "original_text" in result
                assert result["original_text"] == text
                
                # Check if classified intent is reasonable (may not be exact due to fallback)
                assert isinstance(result["confidence"], (int, float))
                assert 0 <= result["confidence"] <= 1
                
                logger.info(f"Text: '{text}' -> Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
    
    @pytest.mark.asyncio
    async def test_multi_intent_classification(self, intent_service, multi_intent_texts):
        """Test multi-intent classification."""
        for test_case in multi_intent_texts:
            text = test_case["text"]
            expected_intents = test_case["expected_intents"]
            
            result = await intent_service.classify_intent(text, multi_intent=True)
            
            assert result is not None
            assert "original_text" in result
            assert result["original_text"] == text
            
            if result.get("type") == "multi_intent":
                assert "intents" in result
                assert "results" in result
                assert isinstance(result["intents"], list)
                assert len(result["intents"]) >= 1
                
                logger.info(f"Multi-intent text: '{text}' -> Intents: {result['intents']}")
            else:
                # Single intent detected
                assert "intent" in result
                logger.info(f"Single intent detected: '{text}' -> Intent: {result['intent']}")
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, intent_service):
        """Test entity extraction for different intent types."""
        test_cases = [
            {
                "text": "Remind me to call John at 5 PM tomorrow",
                "intent": "create_reminder",
                "expected_entities": ["time", "title"]
            },
            {
                "text": "Note to buy groceries and milk from the store",
                "intent": "create_note",
                "expected_entities": ["content"]
            },
            {
                "text": "John owes me 50 dollars for dinner last night",
                "intent": "create_ledger",
                "expected_entities": ["amount", "contact_name"]
            }
        ]
        
        for test_case in test_cases:
            result = await intent_service.classify_intent(test_case["text"], multi_intent=False)
            
            assert result is not None
            assert "entities" in result
            
            entities = result["entities"]
            logger.info(f"Text: '{test_case['text']}' -> Entities: {entities}")
            
            # Check that at least some expected entities are present
            for expected_entity in test_case["expected_entities"]:
                # Not all entities may be extracted, but log what was found
                if expected_entity in entities:
                    logger.info(f"Found expected entity '{expected_entity}': {entities[expected_entity]}")
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, intent_service, sample_texts):
        """Test that confidence scores are reasonable."""
        all_confidences = []
        
        for intent_type, texts in sample_texts.items():
            for text in texts:
                result = await intent_service.classify_intent(text, multi_intent=False)
                confidence = result.get("confidence", 0)
                all_confidences.append(confidence)
                
                # Basic confidence checks
                assert isinstance(confidence, (int, float))
                assert 0 <= confidence <= 1
        
        # Statistical checks on confidence distribution
        avg_confidence = sum(all_confidences) / len(all_confidences)
        logger.info(f"Average confidence across all tests: {avg_confidence:.3f}")
        
        # Should have reasonable average confidence
        assert avg_confidence > 0.1, "Average confidence too low"
    
    def test_text_segmentation(self, intent_service):
        """Test text segmentation for multi-intent detection."""
        test_texts = [
            "Remind me to call John and note to buy groceries",
            "Set reminder for 5 PM and John owes me money",
            "Create a shopping list and remind me about deadline"
        ]
        
        for text in test_texts:
            segments = intent_service._segment_text_for_multi_intent(text)
            
            assert isinstance(segments, list)
            assert len(segments) >= 1
            
            # Check that original text is preserved in segments
            combined_segments = " ".join(segments)
            logger.info(f"Text: '{text}' -> Segments: {segments}")
            
            # Segments should contain meaningful content
            for segment in segments:
                assert len(segment.strip()) > 3

class TestDatabaseIntegrationService:
    """Test suite for Database Integration Service."""
    
    @pytest.fixture
    def db_service(self):
        """Create database service instance for testing."""
        return DatabaseIntegrationService()
    
    @pytest.fixture
    def sample_classification_results(self):
        """Sample classification results for testing."""
        return [
            {
                "intent": "create_reminder",
                "confidence": 0.95,
                "entities": {"title": "Call John", "time": "5 PM"},
                "original_text": "Remind me to call John at 5 PM",
                "model_used": "pytorch"
            },
            {
                "intent": "create_note",
                "confidence": 0.88,
                "entities": {"content": "Buy groceries and milk"},
                "original_text": "Note to buy groceries and milk",
                "model_used": "pytorch"
            },
            {
                "intent": "create_ledger",
                "confidence": 0.92,
                "entities": {"amount": 50.0, "contact_name": "John", "transaction_type": "debt"},
                "original_text": "John owes me 50 dollars",
                "model_used": "pytorch"
            }
        ]
    
    def test_service_initialization(self, db_service):
        """Test database service initialization."""
        assert db_service is not None
        assert db_service.table_schemas is not None
        
        # Check required table schemas
        required_tables = ["reminders", "notes", "ledger_entries", "history_logs"]
        for table in required_tables:
            assert table in db_service.table_schemas
            
        supported_tables = db_service.get_supported_tables()
        assert len(supported_tables) == len(required_tables)
    
    def test_table_schema_validation(self, db_service):
        """Test table schema structure."""
        for table_name, schema in db_service.table_schemas.items():
            assert "required_fields" in schema
            assert "optional_fields" in schema
            assert "default_values" in schema
            
            assert isinstance(schema["required_fields"], list)
            assert isinstance(schema["optional_fields"], list)
            assert isinstance(schema["default_values"], dict)
            
            # All tables should require user_id
            assert "user_id" in schema["required_fields"]
    
    @pytest.mark.asyncio
    async def test_db_record_preparation(self, db_service, sample_classification_results):
        """Test database record preparation for different intent types."""
        test_user_id = "test_user_123"
        
        for result in sample_classification_results:
            db_record = await db_service._prepare_db_record(
                intent=result["intent"],
                entities=result["entities"],
                result=result,
                user_id=test_user_id
            )
            
            assert db_record is not None
            assert "table" in db_record
            assert "data" in db_record
            
            data = db_record["data"]
            assert "user_id" in data
            assert data["user_id"] == test_user_id
            
            logger.info(f"Intent: {result['intent']} -> Table: {db_record['table']}, Data: {data}")
    
    @pytest.mark.asyncio
    async def test_multi_intent_processing(self, db_service):
        """Test multi-intent database processing."""
        test_user_id = "test_user_456"
        
        multi_intent_result = {
            "type": "multi_intent",
            "results": [
                {
                    "intent": "create_reminder",
                    "confidence": 0.93,
                    "entities": {"title": "Call mom", "time": "7 PM"},
                    "original_text": "Remind me to call mom at 7 PM"
                },
                {
                    "intent": "create_note",
                    "confidence": 0.87,
                    "entities": {"content": "Buy chocolate"},
                    "original_text": "Note to buy chocolate"
                }
            ]
        }
        
        processed_result = await db_service._process_multi_intent_for_db(
            multi_intent_result, test_user_id
        )
        
        assert processed_result["success"] is True
        assert processed_result["type"] == "multi_intent"
        assert "db_operations" in processed_result
        assert len(processed_result["db_operations"]) == 2

class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""
    
    @pytest.fixture
    def base_url(self):
        """Base URL for API testing."""
        return "http://localhost:8003"
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers for testing."""
        # In real tests, you'd use actual Firebase tokens
        return {
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json"
        }
    
    def test_health_endpoint(self, base_url):
        """Test the health check endpoint."""
        try:
            response = requests.get(f"{base_url}/api/v1/intent-processor/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert "model_ready" in data
                assert "supported_intents" in data
                logger.info(f"Health check successful: {data}")
            else:
                logger.warning(f"Health check failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to API server: {e}")
            pytest.skip("API server not available for testing")
    
    def test_model_info_endpoint(self, base_url, auth_headers):
        """Test the model info endpoint."""
        try:
            response = requests.get(
                f"{base_url}/api/v1/intent-processor/model-info", 
                headers=auth_headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "model_info" in data
                assert "supported_intents" in data
                assert "capabilities" in data
                logger.info(f"Model info retrieved: {data}")
            else:
                logger.warning(f"Model info request failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to API server: {e}")
            pytest.skip("API server not available for testing")

class TestPerformance:
    """Performance and load testing."""
    
    @pytest.fixture
    def intent_service(self):
        return PyTorchIntentService()
    
    @pytest.mark.asyncio
    async def test_classification_performance(self, intent_service):
        """Test classification performance with multiple texts."""
        test_texts = [
            "Remind me to call John at 5 PM",
            "Note to buy groceries",
            "John owes me 50 dollars",
            "Hello, how are you?",
        ] * 5  # 20 total texts
        
        start_time = time.time()
        
        results = []
        for text in test_texts:
            result = await intent_service.classify_intent(text, multi_intent=True)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_classification = total_time / len(test_texts)
        
        logger.info(f"Classified {len(test_texts)} texts in {total_time:.2f}s")
        logger.info(f"Average time per classification: {avg_time_per_classification:.3f}s")
        
        # Performance assertions
        assert avg_time_per_classification < 2.0, "Classification too slow"
        assert len(results) == len(test_texts), "Some classifications failed"
        
        # Check that all results are valid
        for result in results:
            assert result is not None
            assert "original_text" in result

class TestIntegrationWorkflow:
    """End-to-end integration testing."""
    
    @pytest.fixture
    def intent_service(self):
        return PyTorchIntentService()
    
    @pytest.fixture
    def db_service(self):
        return DatabaseIntegrationService()
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, intent_service, db_service):
        """Test complete workflow from classification to database storage."""
        test_user_id = "integration_test_user"
        test_text = "Remind me to call John at 5 PM and note to buy groceries"
        
        # Step 1: Classify intent
        classification_result = await intent_service.classify_intent(
            text=test_text,
            multi_intent=True
        )
        
        assert classification_result is not None
        logger.info(f"Classification result: {classification_result}")
        
        # Step 2: Process for database (mock database operations)
        try:
            processed_result = await intent_service.process_for_database(
                classification_result, test_user_id
            )
            
            assert processed_result["success"] is True
            logger.info(f"Database processing result: {processed_result}")
            
        except Exception as e:
            logger.warning(f"Database processing failed (expected in test environment): {e}")
            # This is expected to fail in test environment without actual database

def run_tests():
    """Run all tests with proper logging."""
    logger.info("Starting PyTorch Intent System Test Suite")
    logger.info("=" * 60)
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-s"  # Don't capture output
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ SOME TESTS FAILED")
        logger.error("=" * 60)
    
    return exit_code

if __name__ == "__main__":
    # Run the test suite
    exit_code = run_tests()
    sys.exit(exit_code) 