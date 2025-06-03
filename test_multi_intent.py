#!/usr/bin/env python3
"""
Test script for multi-intent processing functionality.
This script demonstrates how the system handles utterances containing multiple intents.
"""

import asyncio
import json
from services.intent_service import IntentService
from services.intent_processor_service import IntentProcessorService

# Test cases for multi-intent processing
TEST_CASES = [
    {
        "name": "Reminder + Note",
        "text": "Set a reminder for 1 a.m. to sleep and set a note to buy chocolate",
        "expected_intents": ["create_reminder", "create_note"]
    },
    {
        "name": "Multiple Reminders",
        "text": "Remind me to call John at 5 PM and also set a reminder to pick up groceries",
        "expected_intents": ["create_reminder", "create_reminder"]
    },
    {
        "name": "Ledger + Note + Reminder",
        "text": "Sarah owes me $50 and note that I need to buy milk plus remind me to call mom",
        "expected_intents": ["create_ledger", "create_note", "create_reminder"]
    },
    {
        "name": "Single Intent (Baseline)",
        "text": "Remind me to take my medication at 8 PM",
        "expected_intents": ["create_reminder"]
    },
    {
        "name": "Complex Multi-Intent",
        "text": "Set a reminder for tomorrow's meeting and create a note about project deadlines and also track that Mike owes me $25",
        "expected_intents": ["create_reminder", "create_note", "create_ledger"]
    }
]

class MultiIntentTester:
    """Test class for multi-intent processing."""
    
    def __init__(self):
        self.intent_service = IntentService()
        self.intent_processor = IntentProcessorService()
    
    async def test_intent_classification(self, text: str, expected_intents: list) -> dict:
        """Test intent classification for a given text."""
        print(f"\n{'='*60}")
        print(f"Testing: '{text}'")
        print(f"Expected intents: {expected_intents}")
        print(f"{'='*60}")
        
        # Test multi-intent classification
        result = await self.intent_service.classify_intent(text, multi_intent=True)
        
        print(f"\nClassification Result:")
        print(f"Original text: {result.get('original_text', 'N/A')}")
        
        if "intents" in result:
            print(f"Multi-intent detected: {len(result['intents'])} intents")
            for i, intent in enumerate(result['intents'], 1):
                print(f"  {i}. Type: {intent.get('type')} (confidence: {intent.get('confidence', 0.0):.2f})")
                print(f"     Segment: '{intent.get('text_segment', 'N/A')}'")
                print(f"     Entities: {intent.get('entities', {})}")
        else:
            print(f"Single intent: {result.get('intent')} (confidence: {result.get('confidence', 0.0):.2f})")
            print(f"Entities: {result.get('entities', {})}")
        
        return result
    
    async def test_intent_processing(self, intent_result: dict, user_id: str = "test_user_123") -> dict:
        """Test intent processing (database operations)."""
        print(f"\n{'-'*40}")
        print("Testing Database Processing:")
        print(f"{'-'*40}")
        
        # Process the intents
        processing_result = await self.intent_processor.process_intent(
            intent_data=intent_result,
            user_id=user_id
        )
        
        print(f"Processing success: {processing_result.get('success', False)}")
        
        if "results" in processing_result:
            # Multi-intent processing
            total = processing_result.get("total_intents", 0)
            successful = processing_result.get("successful_intents", 0)
            print(f"Multi-intent processing: {successful}/{total} intents successful")
            
            for i, result in enumerate(processing_result.get("results", []), 1):
                status = "✓" if result.get("success", False) else "✗"
                intent = result.get("intent", "unknown")
                print(f"  {i}. {status} {intent}")
                if result.get("success", False) and result.get("data"):
                    data = result.get("data", {})
                    if "reminder_id" in data:
                        print(f"     → Reminder: '{data.get('title')}' at {data.get('time', 'no time')}")
                    elif "note_id" in data:
                        print(f"     → Note: '{data.get('content', '')[:50]}...'")
                    elif "ledger_id" in data:
                        print(f"     → Ledger: {data.get('contact_name')} {data.get('direction')} ${data.get('amount')}")
                elif not result.get("success", False):
                    print(f"     → Error: {result.get('error', 'Unknown error')}")
        else:
            # Single intent processing
            intent = processing_result.get("intent", "unknown")
            if processing_result.get("success", False):
                print(f"✓ Single intent processed: {intent}")
                data = processing_result.get("data", {})
                if data:
                    print(f"  Data: {json.dumps(data, indent=2)}")
            else:
                print(f"✗ Single intent failed: {intent}")
                print(f"  Error: {processing_result.get('error', 'Unknown error')}")
        
        return processing_result
    
    async def run_full_test(self, test_case: dict):
        """Run complete test for a test case."""
        print(f"\n{'#'*80}")
        print(f"TEST CASE: {test_case['name']}")
        print(f"{'#'*80}")
        
        # Step 1: Intent Classification
        intent_result = await self.test_intent_classification(
            test_case["text"], 
            test_case["expected_intents"]
        )
        
        # Step 2: Intent Processing
        processing_result = await self.test_intent_processing(intent_result)
        
        # Step 3: Validation
        print(f"\n{'-'*40}")
        print("Validation:")
        print(f"{'-'*40}")
        
        expected_count = len(test_case["expected_intents"])
        
        if "intents" in intent_result:
            actual_count = len(intent_result["intents"])
            print(f"Expected {expected_count} intents, got {actual_count}")
            
            if "results" in processing_result:
                successful_count = processing_result.get("successful_intents", 0)
                print(f"Successfully processed {successful_count}/{actual_count} intents")
                
                if successful_count == actual_count == expected_count:
                    print("✅ TEST PASSED: All intents detected and processed successfully")
                elif successful_count == actual_count:
                    print("⚠️  TEST PARTIAL: All detected intents processed, but count mismatch")
                else:
                    print("❌ TEST FAILED: Some intents failed to process")
            else:
                print("❌ TEST FAILED: Multi-intent not processed correctly")
        else:
            if expected_count == 1:
                if processing_result.get("success", False):
                    print("✅ TEST PASSED: Single intent processed successfully")
                else:
                    print("❌ TEST FAILED: Single intent processing failed")
            else:
                print("❌ TEST FAILED: Expected multiple intents but got single intent")
        
        return {
            "test_case": test_case["name"],
            "classification_result": intent_result,
            "processing_result": processing_result
        }

async def main():
    """Main test function."""
    print("🚀 Starting Multi-Intent Processing Tests")
    print("=" * 80)
    
    tester = MultiIntentTester()
    results = []
    
    for test_case in TEST_CASES:
        try:
            result = await tester.run_full_test(test_case)
            results.append(result)
        except Exception as e:
            print(f"❌ TEST ERROR: {test_case['name']} - {str(e)}")
            results.append({
                "test_case": test_case["name"],
                "error": str(e)
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        test_name = result["test_case"]
        if "error" in result:
            print(f"❌ {test_name}: ERROR - {result['error']}")
        else:
            # Analyze result
            classification = result.get("classification_result", {})
            processing = result.get("processing_result", {})
            
            if "intents" in classification and "results" in processing:
                # Multi-intent test
                detected = len(classification["intents"])
                successful = processing.get("successful_intents", 0)
                print(f"🔄 {test_name}: {detected} detected, {successful} processed")
            elif "intent" in classification and processing.get("success", False):
                # Single intent test
                print(f"✅ {test_name}: Single intent processed successfully")
            else:
                print(f"❌ {test_name}: Processing failed")
    
    print(f"\n🎉 Multi-Intent Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main()) 