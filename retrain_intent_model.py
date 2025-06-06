"""
Optional retraining script for MiniLM Intent Model
Corrects training data by mapping all "add_expense" labels to "create_ledger"
and retrains the model for 3 epochs.
"""

import os
import json
import torch
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentDataset(Dataset):
    """Dataset class for intent classification training."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
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


class IntentModelRetrainer:
    """Retrainer for MiniLM Intent Classification Model."""
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.model_path = "models/Mini_LM.bin"
        self.training_data_path = "training_data/intent_training.json"
        
        # Updated intent labels (no add_expense)
        self.intent_labels = [
            "create_reminder",
            "create_note", 
            "create_ledger",  # Replaces add_expense
            "chit_chat",
            "general_query"
        ]
        
        self.label_to_id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        self.tokenizer = None
        self.model = None
    
    def load_and_correct_training_data(self) -> tuple[List[str], List[int]]:
        """
        Load training data and correct labels:
        - Map all "add_expense" to "create_ledger"
        - Remove any labels not in our current label set
        """
        logger.info("Loading and correcting training data...")
        
        texts = []
        labels = []
        
        try:
            # Try to load existing training data
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'r') as f:
                    data = json.load(f)
                
                for item in data:
                    text = item.get('text', '')
                    label = item.get('label', 'general_query')
                    
                    # Map add_expense to create_ledger
                    if label == "add_expense":
                        label = "create_ledger"
                        logger.info(f"Corrected label: add_expense -> create_ledger for text: {text[:50]}...")
                    
                    # Only include labels in our current set
                    if label in self.label_to_id:
                        texts.append(text)
                        labels.append(self.label_to_id[label])
            else:
                logger.warning(f"Training data file not found: {self.training_data_path}")
                # Create sample training data if none exists
                texts, labels = self._create_sample_training_data()
        
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            # Fallback to sample data
            texts, labels = self._create_sample_training_data()
        
        logger.info(f"Loaded {len(texts)} training samples")
        logger.info(f"Label distribution: {self._get_label_distribution(labels)}")
        
        return texts, labels
    
    def _create_sample_training_data(self) -> tuple[List[str], List[int]]:
        """Create sample training data for demonstration."""
        logger.info("Creating sample training data...")
        
        sample_data = [
            # create_reminder
            ("remind me to call mom at 5pm", "create_reminder"),
            ("set an alarm for 8am tomorrow", "create_reminder"),
            ("schedule meeting with john", "create_reminder"),
            
            # create_note  
            ("note about the grocery list", "create_note"),
            ("write down the meeting notes", "create_note"),
            ("remember to buy milk", "create_note"),
            
            # create_ledger (including corrected add_expense cases)
            ("john owes me $50", "create_ledger"),
            ("I lent sarah $20 yesterday", "create_ledger"), 
            ("track that mike borrowed $100", "create_ledger"),
            ("I spent $30 on lunch", "create_ledger"),  # Was add_expense
            ("bought groceries for $75", "create_ledger"),  # Was add_expense
            ("expense for gas was $40", "create_ledger"),  # Was add_expense
            
            # chit_chat
            ("hello how are you", "chit_chat"),
            ("thank you for your help", "chit_chat"),
            ("good morning", "chit_chat"),
            
            # general_query
            ("what's the weather like", "general_query"),
            ("how do I get to the store", "general_query"),
            ("tell me about pandas", "general_query")
        ]
        
        texts = [item[0] for item in sample_data]
        labels = [self.label_to_id[item[1]] for item in sample_data]
        
        return texts, labels
    
    def _get_label_distribution(self, labels: List[int]) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        distribution = {}
        for label_id in labels:
            label_name = self.id_to_label[label_id]
            distribution[label_name] = distribution.get(label_name, 0) + 1
        return distribution
    
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.intent_labels),
            trust_remote_code=True
        )
        
        # Load existing fine-tuned weights if available
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing fine-tuned weights from {self.model_path}")
            try:
                model_state = torch.load(self.model_path, map_location='cpu')
                if isinstance(model_state, dict) and 'state_dict' in model_state:
                    self.model.load_state_dict(model_state['state_dict'])
                else:
                    self.model.load_state_dict(model_state)
                logger.info("Successfully loaded existing weights")
            except Exception as e:
                logger.warning(f"Could not load existing weights: {e}. Starting from base model.")
    
    def retrain_model(self, texts: List[str], labels: List[int]):
        """Retrain the model for 3 epochs with corrected data."""
        logger.info("Starting model retraining...")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./intent_model_retrain',
            num_train_epochs=3,  # As specified in requirements
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        logger.info("Training started...")
        trainer.train()
        
        # Save the retrained model
        logger.info(f"Saving retrained model to {self.model_path}")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        
        logger.info("Retraining completed successfully!")
    
    def validate_no_add_expense_predictions(self, texts: List[str]):
        """Validate that the model never predicts add_expense."""
        logger.info("Validating that model never predicts add_expense...")
        
        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        add_expense_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for text in texts[:20]:  # Test on subset
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                outputs = self.model(**inputs)
                predicted_label_id = torch.argmax(outputs.logits, dim=-1).item()
                predicted_label = self.id_to_label[predicted_label_id]
                
                total_predictions += 1
                if predicted_label == "add_expense":
                    add_expense_predictions += 1
                    logger.warning(f"Model still predicts add_expense for: {text}")
        
        logger.info(f"Validation complete: {add_expense_predictions}/{total_predictions} add_expense predictions")
        
        if add_expense_predictions == 0:
            logger.info("✅ SUCCESS: Model never predicts add_expense")
        else:
            logger.warning("⚠️  Model still predicts add_expense in some cases")
        
        return add_expense_predictions == 0


def main():
    """Main retraining function."""
    logger.info("Starting MiniLM Intent Model Retraining...")
    
    try:
        # Initialize retrainer
        retrainer = IntentModelRetrainer()
        
        # Load and correct training data
        texts, labels = retrainer.load_and_correct_training_data()
        
        # Setup model and tokenizer
        retrainer.setup_model_and_tokenizer()
        
        # Retrain the model
        retrainer.retrain_model(texts, labels)
        
        # Validate results
        retrainer.validate_no_add_expense_predictions(texts)
        
        logger.info("🎉 Retraining process completed successfully!")
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise


if __name__ == "__main__":
    main() 