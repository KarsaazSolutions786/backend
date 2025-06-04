#!/usr/bin/env python3
"""
Intent Classification Model Training Script
Comprehensive training pipeline for PyTorch-based intent classification.
"""

import os
import sys
import asyncio
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from services.pytorch_intent_service import PyTorchIntentService
from utils.logger import logger

def create_sample_training_data() -> pd.DataFrame:
    """
    Create comprehensive sample training data for intent classification.
    In production, this would load from your actual dataset files.
    """
    
    # Comprehensive training examples for each intent
    training_data = {
        # Create Reminder Examples
        "create_reminder": [
            "Remind me to call John at 5 PM",
            "Set a reminder for my doctor appointment tomorrow at 2 PM",
            "Don't forget to take medicine at 8 AM",
            "Schedule a meeting with the team next Friday",
            "Alert me about the project deadline on Monday",
            "Remind me to pick up groceries after work",
            "Set alarm for workout session at 6 AM",
            "Remember to call mom tonight at 7 PM",
            "Appointment with dentist tomorrow at 2 PM",
            "Remind me to submit the report by Friday 5 PM",
            "Set reminder to water plants this weekend",
            "Don't let me forget the conference call at 3 PM",
            "Remind me to check email before bed",
            "Set a reminder for my birthday party planning",
            "Alert me to renew my driver's license next month",
            "Remind me to backup computer files weekly",
            "Schedule reminder for quarterly review meeting",
            "Don't forget to feed the cat at dinner time",
            "Remind me about the gym membership renewal",
            "Set reminder to pay rent on the 1st of every month"
        ],
        
        # Create Note Examples  
        "create_note": [
            "Note to buy milk and bread from the store",
            "Write down the meeting agenda for tomorrow",
            "Save this idea for the new project",
            "Create a shopping list for weekend groceries",
            "Jot down the phone number for the electrician",
            "Record the discussion points from today's meeting",
            "Make a note about the client feedback",
            "Add to my grocery list: eggs, cheese, and butter",
            "Note about today's brainstorming session",
            "Write down the recipe ingredients for dinner",
            "Create a note about vacation planning ideas",
            "Add note about the book recommendation",
            "Save note about the restaurant we liked",
            "Write down the WiFi password for guests",
            "Note the key points from the training session",
            "Record thoughts about the new product features",
            "Make a note about the gift ideas for mom",
            "Add note about the car maintenance schedule",
            "Write down the exercise routine from the trainer",
            "Note to research vacation destinations for summer"
        ],
        
        # Create Ledger Examples
        "create_ledger": [
            "John owes me 50 dollars for dinner last night",
            "I borrowed 100 dollars from Sarah for rent",
            "Mark paid me back 25 dollars today",
            "Expense for lunch was 15 dollars at the cafe",
            "Sarah owes me money for the concert tickets",
            "I lent 200 dollars to my friend Mike",
            "Paid 80 dollars for gas this week",
            "Mike owes me 30 bucks for the movie tickets",
            "Cost of groceries was 120 dollars yesterday",
            "Borrowed 75 dollars for the taxi ride home",
            "Lisa paid back the 60 dollars she owed",
            "Spent 45 dollars on office supplies",
            "Tom owes me 85 dollars for the group dinner",
            "Lent 150 dollars to my sister for her car repair",
            "Received 40 dollars from Alex for shared utilities",
            "Expense of 95 dollars for new work shoes",
            "Jenny borrowed 70 dollars for her textbooks",
            "Paid back 110 dollars to my dad",
            "Cost of phone bill was 65 dollars this month",
            "Michael owes me 35 dollars for coffee expenses"
        ],
        
        # Chit Chat Examples
        "chit_chat": [
            "Hello, how are you doing today?",
            "What can you help me with?",
            "Thank you for your assistance",
            "Good morning! How's everything?",
            "How's the weather today?",
            "Can you help me with something?",
            "What time is it right now?",
            "Please assist me with this task",
            "Hi there! Nice to meet you",
            "How does this application work?",
            "What features do you have?",
            "Good evening! Hope you're well",
            "Thanks for being so helpful",
            "What's new today?",
            "How can I get started?",
            "Is everything working properly?",
            "What's your favorite feature?",
            "How long have you been helping users?",
            "What other things can you do?",
            "Have a great day!"
        ]
    }
    
    # Add multi-intent examples for training
    multi_intent_examples = [
        ("Remind me to call John at 5 PM and note to buy groceries", "create_reminder"),
        ("Set reminder for meeting tomorrow and John owes me 50 dollars", "create_reminder"),
        ("Create a shopping list for weekend and remind me about the deadline", "create_note"),
        ("Note about project idea and schedule appointment for next week", "create_note"),
        ("I borrowed 100 dollars from Sarah and remind me to pay back", "create_ledger"),
        ("Add note about vacation plans and set reminder for booking flights", "create_note"),
        ("Mark owes me 40 dollars and remind me to ask him tomorrow", "create_ledger"),
        ("Write down meeting notes and set reminder for follow-up call", "create_note"),
        ("Remind me about dentist appointment and note to buy toothpaste", "create_reminder"),
        ("Set reminder for gym session and add note about new workout routine", "create_reminder")
    ]
    
    # Create DataFrame
    data_rows = []
    
    # Add single intent examples
    for intent, examples in training_data.items():
        for example in examples:
            data_rows.append({
                "text": example,
                "intent": intent,
                "type": "single_intent"
            })
    
    # Add multi-intent examples (using primary intent for training)
    for example, primary_intent in multi_intent_examples:
        data_rows.append({
            "text": example,
            "intent": primary_intent,
            "type": "multi_intent"
        })
    
    df = pd.DataFrame(data_rows)
    logger.info(f"Created training dataset with {len(df)} examples")
    logger.info(f"Intent distribution:\n{df['intent'].value_counts()}")
    
    return df

def save_training_data(df: pd.DataFrame, filepath: str):
    """Save training data to file."""
    try:
        # Save as CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Training data saved to {filepath}")
        
        # Also save as JSON for easier loading
        json_filepath = filepath.replace('.csv', '.json')
        df.to_json(json_filepath, orient='records', indent=2)
        logger.info(f"Training data also saved as JSON to {json_filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")

def load_training_data(filepath: str) -> pd.DataFrame:
    """Load training data from file."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        logger.info(f"Loaded training data from {filepath}: {len(df)} examples")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load training data from {filepath}: {e}")
        raise

async def train_model(args):
    """Main training function."""
    logger.info("Starting intent classification model training...")
    
    try:
        # Initialize the service
        logger.info("Initializing PyTorch Intent Service...")
        intent_service = PyTorchIntentService()
        
        # Check if PyTorch is available
        model_info = intent_service.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        if not model_info["pytorch_available"]:
            logger.error("PyTorch is not available. Cannot train model.")
            logger.info("Please install PyTorch and transformers:")
            logger.info("pip install torch transformers scikit-learn")
            return False
        
        # Prepare training data
        if args.data_file and os.path.exists(args.data_file):
            logger.info(f"Loading training data from {args.data_file}")
            df = load_training_data(args.data_file)
        else:
            logger.info("Creating sample training data...")
            df = create_sample_training_data()
            
            # Save the generated data
            os.makedirs("data", exist_ok=True)
            save_training_data(df, "data/intent_training_data.csv")
        
        # Convert DataFrame to lists for training
        texts = df['text'].tolist()
        intents = df['intent'].tolist()
        
        # Map intents to IDs
        label_to_id = intent_service.label_to_id
        labels = [label_to_id[intent] for intent in intents]
        
        logger.info(f"Training data prepared: {len(texts)} examples")
        logger.info(f"Intent distribution: {dict(df['intent'].value_counts())}")
        
        # Start training
        logger.info(f"Starting model training with {args.epochs} epochs...")
        start_time = time.time()
        
        success = await intent_service.train_model(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if success:
            logger.info(f"✅ Model training completed successfully in {training_time:.2f} seconds!")
            logger.info("Model saved to models/pytorch_model.bin")
            
            # Test the trained model
            await test_trained_model(intent_service)
            
            return True
        else:
            logger.error("❌ Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return False

async def test_trained_model(intent_service: PyTorchIntentService):
    """Test the trained model with sample inputs."""
    logger.info("\n" + "="*50)
    logger.info("TESTING TRAINED MODEL")
    logger.info("="*50)
    
    test_examples = [
        "Remind me to call John at 5 PM",
        "Note to buy groceries and milk",
        "John owes me 50 dollars for dinner",
        "Hello, how are you today?",
        "Set reminder for meeting and John owes me money",  # Multi-intent
        "Create a shopping list and remind me about deadline"  # Multi-intent
    ]
    
    for i, example in enumerate(test_examples, 1):
        logger.info(f"\nTest {i}: '{example}'")
        
        # Test single intent classification
        result = await intent_service.classify_intent(example, multi_intent=False)
        logger.info(f"Single intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        
        # Test multi-intent classification
        multi_result = await intent_service.classify_intent(example, multi_intent=True)
        if multi_result.get("type") == "multi_intent":
            intents = multi_result.get("intents", [])
            logger.info(f"Multi-intent detected: {intents}")
        else:
            logger.info(f"Single intent detected: {multi_result['intent']}")

def evaluate_model_performance(intent_service: PyTorchIntentService, test_data: pd.DataFrame):
    """Evaluate model performance on test data."""
    # This would typically involve computing metrics like accuracy, precision, recall, F1-score
    # For now, we'll just log that evaluation would happen here
    logger.info("Model evaluation would be performed here with test dataset")
    logger.info("Metrics to compute: accuracy, precision, recall, F1-score, confusion matrix")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Train Intent Classification Model")
    
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size (default: 16)")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Path to training data file (CSV or JSON)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test existing model without training")
    parser.add_argument("--create-data", action="store_true",
                       help="Only create and save training data")
    
    args = parser.parse_args()
    
    # Set up logging
    logger.info("Intent Classification Model Training")
    logger.info("="*50)
    logger.info(f"Arguments: {vars(args)}")
    
    # Handle different modes
    if args.create_data:
        logger.info("Creating and saving training data...")
        df = create_sample_training_data()
        os.makedirs("data", exist_ok=True)
        save_training_data(df, "data/intent_training_data.csv")
        logger.info("Training data created and saved!")
        return
    
    if args.test_only:
        logger.info("Testing existing model...")
        intent_service = PyTorchIntentService()
        asyncio.run(test_trained_model(intent_service))
        return
    
    # Run training
    success = asyncio.run(train_model(args))
    
    if success:
        logger.info("\n" + "="*50)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info("Next steps:")
        logger.info("1. Test the model with: python train_intent_model.py --test-only")
        logger.info("2. Integrate with your FastAPI backend")
        logger.info("3. Use the /api/v1/intent-processor endpoints")
        sys.exit(0)
    else:
        logger.error("\n" + "="*50)
        logger.error("❌ TRAINING FAILED")
        logger.error("="*50)
        logger.error("Check logs above for error details")
        sys.exit(1)

if __name__ == "__main__":
    main() 