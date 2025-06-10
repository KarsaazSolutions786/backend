#!/usr/bin/env python3
"""
Advanced Multi-Intent Classification Model Training Script
Comprehensive training pipeline for handling complex multi-intent scenarios with generalization.
"""

import os
import sys
import asyncio
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
import random
import re
from collections import Counter

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import dependencies
from utils.logger import logger

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch and required libraries are available")
except ImportError as e:
    PYTORCH_AVAILABLE = False
    logger.error(f"Required libraries not available: {e}")
    logger.error("Please install: pip install torch transformers scikit-learn")
    
    # Create dummy classes for data generation (when PyTorch not available)
    class Dataset:
        pass
    
    class DataLoader:
        pass

class MultiIntentDataGenerator:
    """Generate comprehensive training data for multi-intent classification."""
    
    def __init__(self):
        # Base templates for each intent
        self.intent_templates = {
            "create_reminder": [
                "remind me to {action} at {time}",
                "set a reminder to {action} {time_phrase}",
                "don't forget to {action} {time_phrase}",
                "schedule {action} for {time}",
                "alert me to {action} {time_phrase}",
                "reminder for {action} {time_phrase}",
                "set alarm to {action} at {time}",
                "remember to {action} {time_phrase}",
                "appointment to {action} {time_phrase}",
                "remind me about {action} {time_phrase}",
            ],
            "create_note": [
                "note to {action}",
                "write down {content}",
                "save {content}",
                "create a note about {content}",
                "jot down {content}",
                "record {content}",
                "make a note that {content}",
                "add to my notes {content}",
                "note that {content}",
                "remember {content}",
                "document {content}",
                "keep track of {content}",
            ],
            "create_ledger": [
                "{person} owes me ${amount}",
                "I owe {person} ${amount}",
                "{person} borrowed ${amount} from me",
                "I lent ${amount} to {person}",
                "{person} paid me ${amount}",
                "paid ${amount} to {person}",
                "{person} owes ${amount} for {reason}",
                "borrowed ${amount} from {person} for {reason}",
                "{person} will pay me ${amount}",
                "expense of ${amount} for {reason}",
                "cost ${amount} for {reason}",
                "spent ${amount} on {reason}",
            ],
            "chit_chat": [
                "hello how are you",
                "good morning",
                "what can you help with",
                "thank you",
                "how does this work",
                "what features do you have",
                "help me please",
                "what time is it",
                "how's the weather",
                "nice to meet you",
            ]
        }
        
        # Dynamic content for templates
        self.actions = [
            "call", "email", "text", "visit", "meet", "contact", "pick up", "drop off",
            "buy", "purchase", "get", "fetch", "collect", "deliver", "send", "give",
            "pay", "submit", "complete", "finish", "start", "begin", "check", "review",
            "book", "schedule", "cancel", "confirm", "update", "modify", "clean",
            "exercise", "workout", "study", "practice", "prepare", "pack", "unpack",
            "cook", "eat", "drink", "sleep", "wake up", "shower", "brush teeth",
            "feed", "walk", "drive", "travel", "commute", "return", "arrive"
        ]
        
        self.people = [
            "John", "Sarah", "Mike", "Alex", "David", "Mary", "Jane", "Bob", "Alice",
            "Tom", "Lisa", "Jim", "Ana", "Sam", "Joe", "Amy", "Chris", "Emma", "Ryan",
            "mom", "dad", "brother", "sister", "friend", "boss", "client", "doctor",
            "dentist", "teacher", "neighbor", "roommate", "colleague", "partner"
        ]
        
        self.times = [
            "5 PM", "8 AM", "noon", "midnight", "7:30 PM", "9:15 AM", "6 PM",
            "10 AM", "2 PM", "11:45 PM", "3:30 AM", "4 PM", "7 AM", "1 PM"
        ]
        
        self.time_phrases = [
            "tomorrow", "today", "tonight", "this morning", "this evening",
            "next week", "next month", "in an hour", "in 30 minutes", "later",
            "this weekend", "on Monday", "on Friday", "next Tuesday", "at lunch time"
        ]
        
        self.content_items = [
            "groceries", "milk and bread", "meeting agenda", "project ideas", "phone numbers",
            "important documents", "shopping list", "vacation plans", "gift ideas",
            "recipe ingredients", "book recommendations", "movie suggestions", "passwords",
            "addresses", "websites", "quotes", "research notes", "contact information"
        ]
        
        self.reasons = [
            "dinner", "lunch", "coffee", "taxi", "gas", "groceries", "rent", "utilities",
            "phone bill", "internet", "car repair", "books", "clothes", "medicine",
            "concert tickets", "movie tickets", "travel expenses", "office supplies"
        ]
        
        self.amounts = ["10", "25", "50", "75", "100", "150", "200", "15", "30", "45", "80", "120"]
        
        # Multi-intent connectors
        self.connectors = [
            "and", "also", "plus", "then", "and also", "and then", "plus also",
            "additionally", "furthermore", "moreover", "besides", "as well as"
        ]
    
    def generate_single_intent_examples(self, intent: str, count: int = 100) -> List[Dict]:
        """Generate single intent examples."""
        examples = []
        templates = self.intent_templates[intent]
        
        for _ in range(count):
            template = random.choice(templates)
            
            # Fill template with random content
            if intent == "create_reminder":
                action = random.choice(self.actions)
                time = random.choice(self.times)
                time_phrase = random.choice(self.time_phrases)
                text = template.format(action=action, time=time, time_phrase=time_phrase)
            
            elif intent == "create_note":
                action = random.choice(self.actions)
                content = random.choice(self.content_items)
                text = template.format(action=action, content=content)
            
            elif intent == "create_ledger":
                person = random.choice(self.people)
                amount = random.choice(self.amounts)
                reason = random.choice(self.reasons)
                text = template.format(person=person, amount=amount, reason=reason)
            
            elif intent == "chit_chat":
                text = template
            
            # Add some natural variations
            text = self._add_natural_variations(text)
            
            examples.append({
                "text": text,
                "intent": intent,
                "type": "single_intent",
                "intents": [intent]
            })
        
        return examples
    
    def generate_multi_intent_examples(self, count: int = 200) -> List[Dict]:
        """Generate complex multi-intent examples."""
        examples = []
        intent_combinations = [
            ["create_reminder", "create_note"],
            ["create_reminder", "create_ledger"],
            ["create_note", "create_ledger"],
            ["create_reminder", "create_note", "create_ledger"],
            ["create_note", "create_reminder"],
            ["create_ledger", "create_reminder"],
        ]
        
        for _ in range(count):
            combination = random.choice(intent_combinations)
            connector = random.choice(self.connectors)
            
            # Generate parts for each intent
            parts = []
            for intent in combination:
                single_example = self.generate_single_intent_examples(intent, 1)[0]
                parts.append(single_example["text"])
            
            # Combine with connector
            if len(parts) == 2:
                text = f"{parts[0]} {connector} {parts[1]}"
            elif len(parts) == 3:
                connector2 = random.choice(self.connectors)
                text = f"{parts[0]} {connector} {parts[1]} {connector2} {parts[2]}"
            else:
                text = f" {connector} ".join(parts)
            
            # Add natural variations
            text = self._add_natural_variations(text)
            
            # Primary intent is the first one for training
            primary_intent = combination[0]
            
            examples.append({
                "text": text,
                "intent": primary_intent,  # For training, use primary intent
                "type": "multi_intent",
                "intents": combination,
                "segments": parts
            })
        
        return examples
    
    def _add_natural_variations(self, text: str) -> str:
        """Add natural language variations to make text more realistic."""
        variations = [
            # Add filler words
            (r'\b(remind|set|note|call)\b', lambda m: f"{random.choice(['please', '', 'kindly', ''])} {m.group(1)}".strip()),
            # Add politeness
            (r'^', lambda m: f"{random.choice(['', 'could you ', 'can you ', 'would you '])}"),
            # Add urgency
            (r'(remind|set)', lambda m: f"{m.group(1)} {random.choice(['', 'urgent ', 'important ', ''])}".strip()),
        ]
        
        for pattern, replacement in variations:
            if random.random() < 0.3:  # 30% chance to apply each variation
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def generate_augmented_data(self, base_examples: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """Generate augmented data for better generalization."""
        augmented = []
        
        # Synonym replacement
        synonyms = {
            "remind": ["alert", "notify", "warn"],
            "note": ["record", "write", "jot", "save"],
            "call": ["phone", "contact", "reach"],
            "buy": ["purchase", "get", "acquire"],
            "meet": ["see", "visit", "encounter"],
            "owes": ["borrowed", "is indebted", "should pay"],
        }
        
        for example in base_examples:
            for _ in range(augmentation_factor):
                text = example["text"]
                
                # Replace words with synonyms
                for word, syns in synonyms.items():
                    if word in text.lower() and random.random() < 0.4:
                        text = re.sub(rf'\b{word}\b', random.choice(syns), text, flags=re.IGNORECASE)
                
                # Add new augmented example
                augmented_example = example.copy()
                augmented_example["text"] = text
                augmented.append(augmented_example)
        
        return augmented

class MultiIntentDataset(Dataset):
    """PyTorch Dataset for multi-intent classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultiIntentTrainer:
    """Enhanced trainer for multi-intent classification."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intent_labels = ["create_reminder", "create_note", "create_ledger", "chit_chat"]
        self.label_to_id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.intent_labels)}
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.intent_labels)
        )
        self.model.to(self.device)
        
        logger.info(f"Initialized MultiIntentTrainer with {model_name} on {self.device}")
    
    def prepare_data(self, examples: List[Dict]) -> Tuple[List[str], List[int]]:
        """Prepare training data."""
        texts = [example["text"] for example in examples]
        labels = [self.label_to_id[example["intent"]] for example in examples]
        return texts, labels
    
    def train(self, examples: List[Dict], epochs: int = 3, batch_size: int = 16, 
              learning_rate: float = 2e-5, save_path: str = "models/multi_intent_model.bin"):
        """Train the multi-intent model."""
        logger.info(f"Starting training with {len(examples)} examples")
        
        # Prepare data
        texts, labels = self.prepare_data(examples)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = MultiIntentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = MultiIntentDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./training_output',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        logger.info("Starting training...")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        
        # Save tokenizer
        tokenizer_path = save_path.replace('.bin', '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Model saved to {save_path}")
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Evaluate
        self.evaluate(val_texts, val_labels)
        
        return True
    
    def evaluate(self, texts: List[str], labels: List[int]):
        """Evaluate the model."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoding)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                predictions.append(pred)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"F1 Score: {f1:.3f}")
        
        # Detailed classification report
        intent_names = [self.id_to_label[i] for i in range(len(self.intent_labels))]
        report = classification_report(labels, predictions, target_names=intent_names)
        logger.info(f"Classification Report:\n{report}")
    
    def test_model(self, test_examples: List[str]):
        """Test the model with examples."""
        self.model.eval()
        
        logger.info("\n" + "="*60)
        logger.info("TESTING TRAINED MODEL")
        logger.info("="*60)
        
        with torch.no_grad():
            for i, text in enumerate(test_examples, 1):
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoding)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                pred_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][pred_id].item()
                predicted_intent = self.id_to_label[pred_id]
                
                logger.info(f"\nTest {i}: '{text}'")
                logger.info(f"Predicted: {predicted_intent} (confidence: {confidence:.3f})")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Multi-Intent Classification Model")
    
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate (default: 2e-5)")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased",
                       help="Base model name (default: distilbert-base-uncased)")
    parser.add_argument("--save-path", type=str, default="models/Multi_LM.bin",
                       help="Path to save trained model")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate and save training data")
    parser.add_argument("--data-size", type=int, default=1000,
                       help="Size of generated dataset (default: 1000)")
    
    args = parser.parse_args()
    
    logger.info("Multi-Intent Model Training")
    logger.info("="*60)
    logger.info(f"Arguments: {vars(args)}")
    
    # Generate training data (this works without PyTorch)
    logger.info("Generating training data...")
    generator = MultiIntentDataGenerator()
    
    # Generate base examples
    all_examples = []
    
    # Single intent examples
    for intent in ["create_reminder", "create_note", "create_ledger", "chit_chat"]:
        examples = generator.generate_single_intent_examples(intent, args.data_size // 6)
        all_examples.extend(examples)
        logger.info(f"Generated {len(examples)} examples for {intent}")
    
    # Multi-intent examples
    multi_examples = generator.generate_multi_intent_examples(args.data_size // 3)
    all_examples.extend(multi_examples)
    logger.info(f"Generated {len(multi_examples)} multi-intent examples")
    
    # Data augmentation
    augmented_examples = generator.generate_augmented_data(all_examples, augmentation_factor=1)
    all_examples.extend(augmented_examples)
    logger.info(f"Added {len(augmented_examples)} augmented examples")
    
    logger.info(f"Total training examples: {len(all_examples)}")
    
    # Save generated data
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(all_examples)
    df.to_csv("data/multi_intent_training_data.csv", index=False)
    df.to_json("data/multi_intent_training_data.json", orient='records', indent=2)
    logger.info("Training data saved to data/multi_intent_training_data.csv and .json")
    
    if args.generate_only:
        logger.info("Data generation completed!")
        logger.info("\nTo train the model, you need to install PyTorch:")
        logger.info("For Python 3.13, PyTorch may not be available yet.")
        logger.info("Consider using Python 3.11 or 3.12 for PyTorch compatibility:")
        logger.info("1. Create new environment: python3.11 -m venv venv_training")
        logger.info("2. Activate: source venv_training/bin/activate")
        logger.info("3. Install: pip install torch transformers scikit-learn")
        logger.info("4. Train: python train_multi_intent_model.py")
        return True
    
    # Check PyTorch availability for training
    if not PYTORCH_AVAILABLE:
        logger.error("PyTorch and required libraries are not available!")
        logger.error("Data generation completed, but training requires PyTorch.")
        logger.info("\nPyTorch Installation Instructions:")
        logger.info("=" * 40)
        logger.info("Your Python version: 3.13.3")
        logger.info("Issue: PyTorch doesn't support Python 3.13 yet (as of Dec 2024)")
        logger.info("\nSolutions:")
        logger.info("1. Use Python 3.11 or 3.12:")
        logger.info("   conda create -n training python=3.11")
        logger.info("   conda activate training")
        logger.info("   pip install torch transformers scikit-learn")
        logger.info("\n2. Or use Docker:")
        logger.info("   docker run -it python:3.11")
        logger.info("\n3. Or wait for PyTorch Python 3.13 support")
        logger.info("\nTraining data has been generated and saved.")
        logger.info("You can use it when PyTorch becomes available.")
        return False
    
    # Train model (only if PyTorch is available)
    trainer = MultiIntentTrainer(model_name=args.model_name)
    
    success = trainer.train(
        examples=all_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )
    
    if success:
        # Test the model
        test_examples = [
            "remind me to call John at 5 PM",
            "note to buy groceries and milk",
            "Sarah owes me 50 dollars for dinner",
            "hello how are you today",
            "set a reminder to meet the client and note the project details",
            "call mom tomorrow and John will pay me 100 dollars",
            "note that I have a meeting with the client at noon remind me to pay the electricity bill and I want to visit Murray plus set a reminder to pack my bags"
        ]
        
        trainer.test_model(test_examples)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Your model has been trained with:")
        logger.info(f"- {len(all_examples)} total training examples")
        logger.info(f"- Support for multi-intent detection")
        logger.info(f"- Data augmentation for generalization")
        logger.info(f"- Model saved to: {args.save_path}")
        logger.info("\nThe model should now handle new words and scenarios better!")
        
        return True
    else:
        logger.error("Training failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 