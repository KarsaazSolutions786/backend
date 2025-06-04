#!/usr/bin/env python3

import asyncio
from services.pytorch_intent_service import PyTorchIntentService

async def test_pytorch_multi_intent():
    """Direct test of PyTorch Intent Service multi-intent functionality."""
    
    print("🧪 Testing PyTorch Intent Service Multi-Intent Classification")
    print("=" * 60)
    
    # Initialize the PyTorch Intent Service
    service = PyTorchIntentService()
    
    print(f"📊 Model Info:")
    print(f"   Ready: {service.is_ready()}")
    model_info = service.get_model_info()
    print(f"   Model File: {model_info.get('model_file')}")
    print(f"   PyTorch Available: {model_info.get('pytorch_available')}")
    print(f"   Device: {model_info.get('device')}")
    print(f"   Model Type: {model_info.get('model_type')}")
    
    # Test cases specifically for multi-intent
    test_cases = [
        {
            "text": "Remind me to call John at 5 PM and note to buy groceries",
            "description": "Reminder + Note"
        },
        {
            "text": "Set reminder for meeting tomorrow and John owes me 50 dollars",
            "description": "Reminder + Ledger"
        },
        {
            "text": "Create a note about project and Sarah owes me 30 dollars",
            "description": "Note + Ledger"
        },
        {
            "text": "Sarah owes me $50 and note that I need to buy milk plus remind me to call mom",
            "description": "Ledger + Note + Reminder"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"   Input: '{test_case['text']}'")
        
        # Test multi-intent classification
        result = await service.classify_intent(test_case['text'], multi_intent=True)
        
        print(f"   Result Type: {result.get('type')}")
        print(f"   Model Used: {result.get('model_used', 'unknown')}")
        
        if result.get('type') == 'multi_intent':
            print(f"   ✅ Multi-intent detected!")
            print(f"   Number of Intents: {result.get('num_intents')}")
            print(f"   Intents: {result.get('intents')}")
            print(f"   Overall Confidence: {result.get('overall_confidence', 0):.2f}")
            
            for j, segment_result in enumerate(result.get('results', []), 1):
                print(f"     Segment {j}: '{segment_result.get('segment_text')}'")
                print(f"       Intent: {segment_result.get('intent')} (confidence: {segment_result.get('confidence', 0):.2f})")
                print(f"       Entities: {segment_result.get('entities', {})}")
        elif result.get('type') == 'single_intent':
            print(f"   ⚠️  Single intent detected: {result.get('intent')} (confidence: {result.get('confidence', 0):.2f})")
            print(f"   Entities: {result.get('entities', {})}")
        else:
            print(f"   ❌ Error or unexpected result: {result}")
    
    print("\n" + "=" * 60)
    print("✅ PyTorch Intent Service Test Completed!")

if __name__ == "__main__":
    asyncio.run(test_pytorch_multi_intent()) 