#!/usr/bin/env python3
"""
Demo Script for PyTorch Intent Classification System
Showcases single-intent, multi-intent, and database integration capabilities.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from services.pytorch_intent_service import PyTorchIntentService
from services.database_integration_service import DatabaseIntegrationService
from utils.logger import logger

class PyTorchIntentDemo:
    """Demo class for PyTorch Intent Classification System."""
    
    def __init__(self):
        self.intent_service = PyTorchIntentService()
        self.db_service = DatabaseIntegrationService()
        self.demo_user_id = "demo_user_12345"
    
    def print_banner(self, title: str):
        """Print a formatted banner."""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def print_result(self, text: str, result: Dict[str, Any], elapsed_time: float = None):
        """Print classification result in a formatted way."""
        print(f"\n📝 Input: '{text}'")
        if elapsed_time:
            print(f"⏱️  Time: {elapsed_time:.3f}s")
        
        if result.get("type") == "multi_intent":
            print(f"🎯 Type: Multi-Intent Detection")
            print(f"🔢 Total Intents: {len(result.get('intents', []))}")
            print(f"📊 Overall Confidence: {result.get('overall_confidence', 0):.3f}")
            print(f"🔗 Intents: {', '.join(result.get('intents', []))}")
            
            if result.get('segments'):
                print("📂 Segments:")
                for i, segment in enumerate(result['segments'], 1):
                    print(f"   {i}. '{segment}'")
            
            if result.get('results'):
                print("📋 Detailed Results:")
                for i, res in enumerate(result['results'], 1):
                    intent = res.get('intent', 'unknown')
                    confidence = res.get('confidence', 0)
                    print(f"   {i}. {intent} (confidence: {confidence:.3f})")
                    if res.get('entities'):
                        print(f"      Entities: {res['entities']}")
        else:
            # Single intent
            intent = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0)
            entities = result.get('entities', {})
            
            print(f"🎯 Intent: {intent}")
            print(f"📊 Confidence: {confidence:.3f}")
            print(f"🏷️  Entities: {entities}")
        
        print(f"🤖 Model: {result.get('model_used', 'unknown')}")
    
    async def demo_single_intent_classification(self):
        """Demo single intent classification."""
        self.print_banner("SINGLE INTENT CLASSIFICATION DEMO")
        
        test_cases = [
            "Remind me to call John at 5 PM tomorrow",
            "Create a note to buy groceries and milk",
            "John owes me 50 dollars for dinner last night",
            "Hello, how are you doing today?",
            "Set an alarm for 6 AM workout session",
            "Write down the meeting agenda for next week",
            "Mike paid me back 30 dollars today",
            "What can you help me with?"
        ]
        
        for text in test_cases:
            start_time = time.time()
            result = await self.intent_service.classify_intent(text, multi_intent=False)
            elapsed_time = time.time() - start_time
            
            self.print_result(text, result, elapsed_time)
            await asyncio.sleep(0.5)  # Brief pause for readability
    
    async def demo_multi_intent_classification(self):
        """Demo multi-intent classification."""
        self.print_banner("MULTI-INTENT CLASSIFICATION DEMO")
        
        test_cases = [
            "Remind me to call John at 5 PM and note to buy groceries",
            "Set reminder for meeting tomorrow and John owes me 50 dollars",
            "Create a shopping list for weekend and remind me about the deadline",
            "Note about project idea and schedule appointment for next week",
            "I borrowed 100 dollars from Sarah and remind me to pay back",
            "Add note about vacation plans and set reminder for booking flights",
            "Mark owes me 40 dollars and remind me to ask him tomorrow",
            "Write down meeting notes and set reminder for follow-up call",
            "Hello there and remind me about dentist appointment",
            "Set reminder for gym session and add note about new workout routine"
        ]
        
        for text in test_cases:
            start_time = time.time()
            result = await self.intent_service.classify_intent(text, multi_intent=True)
            elapsed_time = time.time() - start_time
            
            self.print_result(text, result, elapsed_time)
            await asyncio.sleep(0.5)  # Brief pause for readability
    
    async def demo_entity_extraction(self):
        """Demo entity extraction capabilities."""
        self.print_banner("ENTITY EXTRACTION DEMO")
        
        test_cases = [
            {
                "text": "Remind me to call Dr. Smith at 3:30 PM on Friday",
                "expected_entities": ["contact", "time", "date"]
            },
            {
                "text": "Note to buy 2 dozen eggs, milk, and bread from Walmart",
                "expected_entities": ["items", "quantity", "location"]
            },
            {
                "text": "Sarah owes me $75.50 for the concert tickets from last weekend",
                "expected_entities": ["amount", "contact", "reason", "time"]
            },
            {
                "text": "Create a shopping list with apples, bananas, chicken, and rice",
                "expected_entities": ["items", "list_type"]
            }
        ]
        
        for test_case in test_cases:
            text = test_case["text"]
            expected = test_case["expected_entities"]
            
            result = await self.intent_service.classify_intent(text, multi_intent=False)
            entities = result.get("entities", {})
            
            print(f"\n📝 Input: '{text}'")
            print(f"🎯 Intent: {result.get('intent')}")
            print(f"🔍 Expected Entities: {expected}")
            print(f"🏷️  Extracted Entities: {entities}")
            
            # Check which expected entities were found
            found_entities = []
            for entity_type in expected:
                if any(entity_type.lower() in key.lower() for key in entities.keys()):
                    found_entities.append(entity_type)
            
            if found_entities:
                print(f"✅ Found: {found_entities}")
            else:
                print("❌ No expected entities found")
            
            await asyncio.sleep(0.3)
    
    async def demo_database_integration(self):
        """Demo database integration."""
        self.print_banner("DATABASE INTEGRATION DEMO")
        
        print(f"🆔 Demo User ID: {self.demo_user_id}")
        
        # Test cases with expected database tables
        test_cases = [
            {
                "text": "Remind me to call the doctor tomorrow at 2 PM",
                "expected_table": "reminders"
            },
            {
                "text": "Note to research vacation destinations for summer",
                "expected_table": "notes"
            },
            {
                "text": "Tom owes me 85 dollars for the group dinner",
                "expected_table": "ledger_entries"
            },
            {
                "text": "Thanks for your help today!",
                "expected_table": "history_logs"
            }
        ]
        
        for test_case in test_cases:
            text = test_case["text"]
            expected_table = test_case["expected_table"]
            
            print(f"\n📝 Processing: '{text}'")
            print(f"📊 Expected Table: {expected_table}")
            
            # Step 1: Classify intent
            classification_result = await self.intent_service.classify_intent(text, multi_intent=False)
            intent = classification_result.get("intent")
            confidence = classification_result.get("confidence", 0)
            
            print(f"🎯 Classified Intent: {intent} (confidence: {confidence:.3f})")
            
            # Step 2: Prepare for database (simulate)
            try:
                db_record = await self.db_service._prepare_db_record(
                    intent=intent,
                    entities=classification_result.get("entities", {}),
                    result=classification_result,
                    user_id=self.demo_user_id
                )
                
                if db_record:
                    actual_table = db_record["table"]
                    data = db_record["data"]
                    
                    print(f"🗄️  Actual Table: {actual_table}")
                    print(f"✅ Table Match: {'Yes' if actual_table == expected_table else 'No'}")
                    print(f"📋 Database Record Preview:")
                    for key, value in data.items():
                        if key != "user_id":  # Skip user_id for brevity
                            print(f"   {key}: {value}")
                else:
                    print("❌ Failed to prepare database record")
                    
            except Exception as e:
                print(f"❌ Database preparation error: {e}")
            
            await asyncio.sleep(0.5)
    
    async def demo_performance_benchmark(self):
        """Demo performance benchmarking."""
        self.print_banner("PERFORMANCE BENCHMARK DEMO")
        
        test_texts = [
            "Remind me to call John",
            "Note to buy groceries",
            "Sarah owes me money",
            "Hello there",
        ] * 5  # 20 total tests
        
        print(f"🧪 Running performance test with {len(test_texts)} classifications...")
        
        # Single intent performance
        start_time = time.time()
        single_results = []
        
        for text in test_texts:
            result = await self.intent_service.classify_intent(text, multi_intent=False)
            single_results.append(result)
        
        single_total_time = time.time() - start_time
        single_avg_time = single_total_time / len(test_texts)
        
        print(f"📊 Single Intent Classification:")
        print(f"   Total Time: {single_total_time:.2f}s")
        print(f"   Average Time: {single_avg_time:.3f}s per classification")
        print(f"   Throughput: {len(test_texts)/single_total_time:.1f} classifications/second")
        
        # Multi intent performance
        start_time = time.time()
        multi_results = []
        
        for text in test_texts:
            result = await self.intent_service.classify_intent(text, multi_intent=True)
            multi_results.append(result)
        
        multi_total_time = time.time() - start_time
        multi_avg_time = multi_total_time / len(test_texts)
        
        print(f"📊 Multi-Intent Classification:")
        print(f"   Total Time: {multi_total_time:.2f}s")
        print(f"   Average Time: {multi_avg_time:.3f}s per classification")
        print(f"   Throughput: {len(test_texts)/multi_total_time:.1f} classifications/second")
        
        # Confidence analysis
        single_confidences = [r.get("confidence", 0) for r in single_results if r.get("confidence")]
        if single_confidences:
            avg_confidence = sum(single_confidences) / len(single_confidences)
            min_confidence = min(single_confidences)
            max_confidence = max(single_confidences)
            
            print(f"📈 Confidence Statistics:")
            print(f"   Average: {avg_confidence:.3f}")
            print(f"   Range: {min_confidence:.3f} - {max_confidence:.3f}")
    
    async def demo_model_info(self):
        """Demo model information display."""
        self.print_banner("MODEL INFORMATION DEMO")
        
        model_info = self.intent_service.get_model_info()
        
        print("🤖 Model Configuration:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        print(f"\n🏷️  Supported Intents:")
        for i, intent in enumerate(self.intent_service.intent_labels, 1):
            print(f"   {i}. {intent}")
        
        print(f"\n🗄️  Database Tables:")
        supported_tables = self.db_service.get_supported_tables()
        for i, table in enumerate(supported_tables, 1):
            schema = self.db_service.get_table_schema(table)
            required_fields = schema["required_fields"]
            print(f"   {i}. {table} (required: {', '.join(required_fields)})")
    
    async def demo_error_handling(self):
        """Demo error handling and fallback mechanisms."""
        self.print_banner("ERROR HANDLING & FALLBACK DEMO")
        
        test_cases = [
            "",  # Empty text
            "   ",  # Whitespace only
            "a" * 1000,  # Very long text
            "🎉🔥💯🚀✨",  # Emojis only
            "Ünïcödë tëxt wïth spëcïäl chäräctërs",  # Unicode
            None  # None input (will cause error)
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {repr(text)}")
            
            try:
                if text is None:
                    # This should cause an error
                    result = await self.intent_service.classify_intent(text, multi_intent=False)
                else:
                    result = await self.intent_service.classify_intent(text, multi_intent=False)
                
                if result:
                    intent = result.get("intent", "unknown")
                    confidence = result.get("confidence", 0)
                    model_used = result.get("model_used", "unknown")
                    
                    print(f"✅ Success: {intent} (confidence: {confidence:.3f}, model: {model_used})")
                else:
                    print("❌ No result returned")
                    
            except Exception as e:
                print(f"❌ Error handled: {e}")
                print("🔄 Fallback mechanisms should have activated")
    
    async def run_complete_demo(self):
        """Run the complete demo suite."""
        print("🎉 Welcome to the PyTorch Intent Classification System Demo!")
        print("This demo showcases all the key capabilities of the system.")
        
        # Check if system is ready
        model_info = self.intent_service.get_model_info()
        if not model_info.get("pytorch_available"):
            print("\n⚠️  WARNING: PyTorch not available - using fallback mode")
        
        try:
            await self.demo_model_info()
            await self.demo_single_intent_classification()
            await self.demo_multi_intent_classification()
            await self.demo_entity_extraction()
            await self.demo_database_integration()
            await self.demo_performance_benchmark()
            await self.demo_error_handling()
            
            self.print_banner("DEMO COMPLETED SUCCESSFULLY! 🎉")
            print("The PyTorch Intent Classification System is working correctly.")
            print("\nNext steps:")
            print("1. Train your own model: python train_intent_model.py")
            print("2. Run tests: python test_pytorch_intent_system.py")
            print("3. Start the API server: python main.py")
            print("4. Check the setup guide: PYTORCH_INTENT_SETUP_GUIDE.md")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Demo failed with error: {e}")
            print("Check the logs for more details")

async def main():
    """Main function to run the demo."""
    demo = PyTorchIntentDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    # Run the demo
    print("Starting PyTorch Intent Classification Demo...")
    asyncio.run(main()) 