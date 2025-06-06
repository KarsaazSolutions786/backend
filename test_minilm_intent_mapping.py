"""
Unit tests for MiniLM Intent Service to ensure:
1. No output ever contains "add_expense" 
2. All "add_expense" predictions are mapped to "create_ledger"
3. Duplicate intents are removed while preserving order
"""

import pytest
import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.minilm_intent_service import MiniLMIntentService


class TestMiniLMIntentMapping:
    """Test suite for intent mapping and deduplication."""
    
    @pytest.fixture
    def intent_service(self):
        """Create a MiniLM intent service instance."""
        return MiniLMIntentService()
    
    def test_post_process_intents_mapping(self, intent_service):
        """Test that add_expense is correctly mapped to create_ledger."""
        # Test single intent mapping
        result = intent_service._post_process_intents(["add_expense"])
        assert result == ["create_ledger"]
        
        # Test multiple intents with mapping
        result = intent_service._post_process_intents([
            "create_reminder", 
            "add_expense", 
            "general_query"
        ])
        assert result == ["create_reminder", "create_ledger", "general_query"]
        assert "add_expense" not in result
    
    def test_post_process_intents_deduplication(self, intent_service):
        """Test that duplicate intents are removed while preserving order."""
        # Test basic deduplication
        result = intent_service._post_process_intents([
            "create_reminder", 
            "general_query", 
            "create_reminder"
        ])
        assert result == ["create_reminder", "general_query"]
        assert len(result) == 2
        
        # Test deduplication with mapping
        result = intent_service._post_process_intents([
            "create_reminder",
            "add_expense", 
            "general_query",
            "add_expense",
            "general_query"
        ])
        assert result == ["create_reminder", "create_ledger", "general_query"]
        assert "add_expense" not in result
        assert len(result) == 3
    
    def test_post_process_intents_empty_list(self, intent_service):
        """Test handling of empty intent list."""
        result = intent_service._post_process_intents([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_single_intent_never_returns_add_expense(self, intent_service):
        """Test that single intent classification never returns add_expense."""
        # Test various expense-related texts
        expense_texts = [
            "I spent $50 on groceries",
            "bought coffee for $5",
            "paid the restaurant bill",
            "expense for lunch today",
            "cost me $100 at the store"
        ]
        
        for text in expense_texts:
            result = await intent_service.classify_intent(text, multi_intent=False)
            
            # Ensure no add_expense in single intent result
            assert result.get("intent") != "add_expense", f"Found add_expense for text: {text}"
            
            # The intent should be valid (any intent except add_expense is acceptable)
            valid_intents = ["create_reminder", "create_note", "create_ledger", "chit_chat", "general_query"]
            assert result.get("intent") in valid_intents, f"Invalid intent returned: {result.get('intent')}"
    
    @pytest.mark.asyncio
    async def test_multi_intent_never_returns_add_expense(self, intent_service):
        """Test that multi-intent classification never returns add_expense."""
        # Test texts that might produce add_expense
        multi_intent_texts = [
            "remind me to call mom and I spent $50 on groceries",
            "note about meeting and expense for lunch",
            "set alarm for 8am and bought coffee for $5",
            "create reminder for doctor appointment and paid the bill"
        ]
        
        for text in multi_intent_texts:
            result = await intent_service.classify_intent(text, multi_intent=True)
            
            # Ensure no add_expense in intents list
            intents = result.get("intents", [])
            assert "add_expense" not in intents
            
            # Verify no duplicates
            assert len(intents) == len(set(intents))
    
    @pytest.mark.asyncio
    async def test_rule_based_classification_mapping(self, intent_service):
        """Test that rule-based classification applies mapping correctly."""
        # Force rule-based classification by using simple text
        # that would trigger add_expense patterns
        expense_text = "I spent money on food"
        
        # Get classification result
        result = await intent_service.classify_intent(expense_text, multi_intent=False)
        
        # Should never return add_expense
        assert result.get("intent") != "add_expense"
    
    @pytest.mark.asyncio
    async def test_fallback_response_never_add_expense(self, intent_service):
        """Test that fallback responses never contain add_expense."""
        # Test fallback for single intent
        fallback_single = intent_service._create_fallback_response("test", False)
        assert fallback_single.get("intent") != "add_expense"
        
        # Test fallback for multi intent
        fallback_multi = intent_service._create_fallback_response("test", True)
        intents = fallback_multi.get("intents", [])
        assert "add_expense" not in intents
    
    def test_example_bad_payload_fix(self, intent_service):
        """Test the specific example from requirements."""
        # Simulate the bad payload scenario
        bad_intents = [
            "create_reminder",
            "add_expense",          # Should become create_ledger
            "general_query", 
            "general_query"         # Duplicate should be removed
        ]
        
        # Process the intents
        result = intent_service._post_process_intents(bad_intents)
        
        # Expected result should be deduplicated and mapped
        expected = ["create_reminder", "create_ledger", "general_query"]
        assert result == expected
        assert "add_expense" not in result
        assert len(result) == 3  # No duplicates
    
    @pytest.mark.asyncio
    async def test_comprehensive_no_add_expense_output(self, intent_service):
        """Comprehensive test to ensure no method can output add_expense."""
        test_cases = [
            # Single intent cases
            ("remind me to call john", False),
            ("I spent $100 on shoes", False),
            ("note about the meeting", False),
            ("john owes me $50", False),
            
            # Multi-intent cases  
            ("remind me to call and I spent money", True),
            ("note about meeting and john owes $20", True),
            ("set alarm and bought groceries", True)
        ]
        
        for text, multi_intent in test_cases:
            result = await intent_service.classify_intent(text, multi_intent)
            
            if multi_intent:
                intents = result.get("intents", [])
                assert "add_expense" not in intents, f"Found add_expense in multi-intent result for: {text}"
                # Verify no duplicates
                assert len(intents) == len(set(intents)), f"Found duplicates in: {intents}"
            else:
                intent = result.get("intent", "")
                assert intent != "add_expense", f"Found add_expense in single-intent result for: {text}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 