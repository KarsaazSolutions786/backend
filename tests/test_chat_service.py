#!/usr/bin/env python3
"""
Tests for ChatService with fine-tuned Bloom-560M model
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chat_service import ChatService


class TestChatService:
    """Test cases for ChatService functionality"""
    
    @pytest_asyncio.fixture
    async def chat_service(self):
        """Create a ChatService instance for testing"""
        service = ChatService()
        await asyncio.sleep(2)  # Give it time to initialize
        return service
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, chat_service):
        """Test that ChatService initializes correctly"""
        assert chat_service is not None
        assert hasattr(chat_service, 'model_ready')
        assert hasattr(chat_service, 'model_info')
        
        # Check if service reports as ready
        is_ready = await chat_service.is_ready()
        assert isinstance(is_ready, bool)
        
        # If model is ready, check model info
        if chat_service.model_ready:
            assert 'model_name' in chat_service.model_info
            assert 'mode' in chat_service.model_info
    
    @pytest.mark.asyncio
    async def test_greeting_response(self, chat_service):
        """Test that greeting prompts return polite, non-empty responses."""
        test_user_id = "test_user_123"
        greeting_prompts = [
            "hello there",
            "hey how are you",
            "hi",
            "good morning"
        ]
        
        for prompt in greeting_prompts:
            response = await chat_service.generate_response(prompt, test_user_id)
            
            # Check that response is not empty
            assert response is not None
            assert len(response.strip()) > 0
            
            # Check for polite/helpful language
            response_lower = response.lower()
            politeness_indicators = [
                'hello', 'hi', 'hey', 'great', 'good', 'help', 
                'assist', 'thanks', 'thank you', 'welcome'
            ]
            
            has_polite_language = any(indicator in response_lower for indicator in politeness_indicators)
            assert has_polite_language, f"Response '{response}' should contain polite language"
    
    @pytest.mark.asyncio
    async def test_reminder_request_response(self, chat_service):
        """Test that reminder requests contain questions or helpful responses."""
        test_user_id = "test_user_456"
        reminder_prompts = [
            "set a reminder",
            "remind me",
            "create a reminder",
            "add a reminder"
        ]
        
        for prompt in reminder_prompts:
            response = await chat_service.generate_response(prompt, test_user_id)
            
            # Check that response is not empty
            assert response is not None
            assert len(response.strip()) > 0
            
            # Check for question marks or helpful words
            has_question = '?' in response
            helpful_words = ['what', 'when', 'where', 'time', 'sure', 'help', 'like', 'want']
            has_helpful_words = any(word in response.lower() for word in helpful_words)
            
            assert has_question or has_helpful_words, \
                f"Response '{response}' should contain a question or helpful language"
    
    @pytest.mark.asyncio
    async def test_model_info(self, chat_service):
        """Test that model info is properly set."""
        model_info = chat_service.get_model_info()
        
        assert model_info is not None
        assert isinstance(model_info, dict)
        
        # Check if we're using the fine-tuned model or fallback
        if 'model_name' in model_info:
            expected_models = [
                'bloom-560m-chat',  # Fine-tuned model
                'bloom-560m',       # Base model
                'bloom-560m-enhanced-fallback',  # Fallback mode
                'bloom-560m-fallback'  # Minimal mode
            ]
            assert model_info['model_name'] in expected_models
    
    @pytest.mark.asyncio
    async def test_context_aware_responses(self, chat_service):
        """Test that the service provides context-aware responses."""
        test_user_id = "test_user_789"
        
        # Test with intent context
        context_with_intent = {
            "intent": "create_reminder",
            "entities": {"time": "5 PM"},
            "confidence": 0.95
        }
        
        response = await chat_service.generate_response(
            "remind me to call John", 
            test_user_id, 
            context_with_intent
        )
        
        assert response is not None
        assert len(response.strip()) > 0
        
        # Should acknowledge the reminder creation
        response_lower = response.lower()
        reminder_keywords = ['reminder', 'remind', 'call', 'john', 'created', 'set']
        has_relevant_content = any(keyword in response_lower for keyword in reminder_keywords)
        assert has_relevant_content, f"Response should acknowledge reminder context: '{response}'"
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, chat_service):
        """Test that conversation history is maintained."""
        test_user_id = "test_user_history"
        
        # First message
        response1 = await chat_service.generate_response("hello", test_user_id)
        assert response1 is not None
        
        # Second message - should maintain context
        response2 = await chat_service.generate_response("how are you?", test_user_id)
        assert response2 is not None
        
        # Check that history is being tracked
        history = chat_service._get_conversation_history(test_user_id)
        assert len(history) >= 2  # Should have at least 2 interactions
    
    @pytest.mark.asyncio
    async def test_fallback_behavior(self, chat_service):
        """Test that fallback responses work when model is unavailable."""
        test_user_id = "test_user_fallback"
        
        # Mock model unavailability
        original_model_ready = chat_service.model_ready
        chat_service.model_ready = False
        
        try:
            response = await chat_service.generate_response("hello", test_user_id)
            
            assert response is not None
            assert len(response.strip()) > 0
            
            # Should still be helpful even in fallback mode
            response_lower = response.lower()
            helpful_indicators = ['help', 'assist', 'can', 'here', 'hello', 'hi']
            has_helpful_content = any(indicator in response_lower for indicator in helpful_indicators)
            assert has_helpful_content, f"Fallback response should be helpful: '{response}'"
            
        finally:
            # Restore original state
            chat_service.model_ready = original_model_ready
    
    @pytest.mark.asyncio
    async def test_health_check(self, chat_service):
        """Test the health check functionality."""
        health = await chat_service.health_check()
        
        assert health is not None
        assert isinstance(health, dict)
        assert 'status' in health
        
        # Status should be either healthy or degraded
        assert health['status'] in ['healthy', 'degraded', 'unavailable']
    
    @pytest.mark.asyncio
    async def test_empty_message_handling(self, chat_service):
        """Test handling of empty or invalid messages."""
        test_user_id = "test_user_empty"
        
        empty_messages = ["", "   ", None]
        
        for message in empty_messages:
            try:
                response = await chat_service.generate_response(message or "", test_user_id)
                # Should handle gracefully and return a fallback response
                assert response is not None
                assert len(response.strip()) > 0
            except Exception as e:
                # Should not raise exceptions for empty messages
                pytest.fail(f"Service should handle empty message gracefully, but raised: {e}")
    
    @pytest.mark.asyncio
    async def test_long_message_handling(self, chat_service):
        """Test handling of very long messages."""
        test_user_id = "test_user_long"
        
        # Create a very long message
        long_message = "Can you help me with this very long request? " * 50
        
        response = await chat_service.generate_response(long_message, test_user_id)
        
        assert response is not None
        assert len(response.strip()) > 0
        
        # Response should be reasonable length (not truncated severely)
        assert len(response) > 10


if __name__ == "__main__":
    # Allow direct running of tests
    pytest.main([__file__, "-v"]) 