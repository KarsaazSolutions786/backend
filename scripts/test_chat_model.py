#!/usr/bin/env python3
"""
Simple test script for the fine-tuned Bloom-560M chat model.
This can be run independently to verify model behavior.

Usage:
    python scripts/test_chat_model.py
"""

import sys
import os
import asyncio

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.chat_service import ChatService
from utils.logger import logger

async def test_greeting_responses():
    """Test greeting responses."""
    print("\n🧪 Testing greeting responses...")
    
    chat_service = ChatService()
    test_user_id = "test_user_greetings"
    
    greeting_prompts = [
        "hello there",
        "hey how are you", 
        "hi",
        "good morning"
    ]
    
    for prompt in greeting_prompts:
        try:
            response = await chat_service.generate_response(prompt, test_user_id)
            
            print(f"  Input: '{prompt}'")
            print(f"  Output: '{response}'")
            
            # Check basic requirements
            is_non_empty = len(response.strip()) > 0
            has_polite_language = any(word in response.lower() for word in 
                                    ['hello', 'hi', 'hey', 'great', 'good', 'help', 'assist'])
            
            status = "✅ PASS" if (is_non_empty and has_polite_language) else "❌ FAIL"
            print(f"  Status: {status}")
            print()
            
        except Exception as e:
            print(f"  Input: '{prompt}'")
            print(f"  Error: {e}")
            print(f"  Status: ❌ FAIL")
            print()

async def test_reminder_requests():
    """Test reminder request responses."""
    print("\n🧪 Testing reminder request responses...")
    
    chat_service = ChatService()
    test_user_id = "test_user_reminders"
    
    reminder_prompts = [
        "set a reminder",
        "remind me",
        "create a reminder",
        "add a reminder"
    ]
    
    for prompt in reminder_prompts:
        try:
            response = await chat_service.generate_response(prompt, test_user_id)
            
            print(f"  Input: '{prompt}'")
            print(f"  Output: '{response}'")
            
            # Check requirements
            is_non_empty = len(response.strip()) > 0
            has_question = '?' in response
            has_helpful_words = any(word in response.lower() for word in 
                                  ['what', 'when', 'where', 'time', 'sure', 'help', 'like', 'want'])
            
            status = "✅ PASS" if (is_non_empty and (has_question or has_helpful_words)) else "❌ FAIL"
            print(f"  Status: {status}")
            print()
            
        except Exception as e:
            print(f"  Input: '{prompt}'")
            print(f"  Error: {e}")
            print(f"  Status: ❌ FAIL")
            print()

async def test_model_info():
    """Test model information."""
    print("\n🧪 Testing model information...")
    
    chat_service = ChatService()
    
    try:
        model_info = chat_service.get_model_info()
        
        print(f"  Model Info: {model_info}")
        
        has_model_name = 'model_name' in model_info
        has_mode = 'mode' in model_info or 'error' in model_info
        
        status = "✅ PASS" if (has_model_name or has_mode) else "❌ FAIL"
        print(f"  Status: {status}")
        
        # Check if fine-tuned model is being used
        if 'model_name' in model_info:
            model_name = model_info['model_name']
            if 'chat' in model_name:
                print(f"  ✅ Using fine-tuned model: {model_name}")
            else:
                print(f"  ⚠️  Using base model: {model_name}")
        
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Status: ❌ FAIL")

async def test_health_check():
    """Test health check."""
    print("\n🧪 Testing health check...")
    
    chat_service = ChatService()
    
    try:
        health = await chat_service.health_check()
        
        print(f"  Health Status: {health}")
        
        has_status = 'status' in health
        valid_status = health.get('status') in ['healthy', 'degraded', 'unavailable'] if has_status else False
        
        status = "✅ PASS" if (has_status and valid_status) else "❌ FAIL"
        print(f"  Status: {status}")
        
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Status: ❌ FAIL")

async def main():
    """Run all tests."""
    print("🚀 Testing Fine-tuned Bloom-560M Chat Model")
    print("=" * 50)
    
    # Initialize logging
    logger.info("Starting chat model tests...")
    
    # Run tests
    await test_model_info()
    await test_health_check()
    await test_greeting_responses()
    await test_reminder_requests()
    
    print("\n✅ Testing completed!")
    print("\nNote: If tests are failing, consider:")
    print("1. Running fine-tuning script: python scripts/finetune_bloom.py")
    print("2. Ensuring Models/Bloom_560M_chat exists")
    print("3. Checking that the chat service initializes properly")

if __name__ == "__main__":
    asyncio.run(main()) 