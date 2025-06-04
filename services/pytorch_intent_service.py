"""
PyTorch-based Intent Classification Service with Multi-Intent Support
Handles training, fine-tuning, classification, and database integration.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re
from datetime import datetime
import asyncio

# Core dependencies
from core.config import settings
from utils.logger import logger

# Optional PyTorch imports with graceful fallback
PYTORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available for intent classification")
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library is available")
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn is available")
except ImportError as e:
    logger.warning(f"Scikit-learn not available: {e}")

# Fallback classes for when PyTorch is not available
if not PYTORCH_AVAILABLE:
    class Dataset:
        """Fallback Dataset class when PyTorch is not available."""
        pass
    
    class DataLoader:
        """Fallback DataLoader class when PyTorch is not available."""
        pass
    
    class nn:
        """Fallback nn module when PyTorch is not available."""
        class Module:
            pass

if PYTORCH_AVAILABLE:
    class IntentDataset(Dataset):
        """Custom Dataset for intent classification."""
        
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
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

    class MultiIntentClassifier(nn.Module):
        """Custom multi-intent classifier based on transformer models."""
        
        def __init__(self, model_name: str, num_labels: int, hidden_dropout: float = 0.1):
            super().__init__()
            self.config = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                hidden_dropout_prob=hidden_dropout
            )
            self.dropout = nn.Dropout(hidden_dropout)
            
            # Multi-intent detection head
            self.multi_intent_classifier = nn.Linear(self.config.hidden_size, 2)  # Binary: single/multi
            
        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get hidden states for multi-intent detection
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
            
            return outputs
else:
    # Fallback classes when PyTorch is not available
    class IntentDataset:
        """Fallback IntentDataset class."""
        def __init__(self, *args, **kwargs):
            pass
    
    class MultiIntentClassifier:
        """Fallback MultiIntentClassifier class."""
        def __init__(self, *args, **kwargs):
            pass

class PyTorchIntentService:
    """
    Comprehensive PyTorch-based intent classification service with:
    - Model training and fine-tuning
    - Multi-intent detection and classification
    - Database integration
    - Performance optimization
    """
    
    def __init__(self):
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.model_path = Path("models/pytorch_intent_model")
        self.model_file = Path("models/pytorch_model.bin")
        
        # Intent labels matching your requirements
        self.intent_labels = [
            "create_reminder",  # 0
            "create_note",      # 1  
            "create_ledger",    # 2
            "chit_chat"         # 3
        ]
        
        self.label_to_id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.intent_labels)}
        
        # Multi-intent patterns for text segmentation
        self.multi_intent_patterns = [
            r'\band\s+(?:also\s+)?(?:set|create|add|make|remind|schedule|note)',
            r'\balso\s+(?:set|create|add|make|remind|schedule|note)',
            r'\bthen\s+(?:set|create|add|make|remind|schedule|note)',
            r'\bplus\s+(?:set|create|add|make|remind|schedule|note)',
            r'\band\s+(?:[A-Z][a-z]+\s+)?(?:owes?|owed?|borrowed?|lent)',
            r'(?:\.|;)\s+(?:Set|Create|Add|Make|Remind|Schedule|Note)',
        ]
        
        # Initialize the service
        self._initialize()
    
    def _get_device(self) -> str:
        """Get the best available device for PyTorch."""
        if not PYTORCH_AVAILABLE:
            return "cpu"
        
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU for PyTorch computations")
        
        return device
    
    def _initialize(self):
        """Initialize the intent classification service."""
        try:
            if PYTORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
                self._load_or_initialize_model()
            else:
                logger.warning("PyTorch/Transformers not available. Using fallback classification.")
                self._initialize_fallback_classifier()
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch intent service: {e}")
            self._initialize_fallback_classifier()
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize new one."""
        try:
            if self.model_file.exists():
                logger.info(f"Loading existing PyTorch model from {self.model_file}")
                self._load_model_from_file()
            else:
                logger.info("Initializing new model for training")
                self._initialize_new_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialize_new_model()
    
    def _load_model_from_file(self):
        """Load the trained model from pytorch_model.bin."""
        try:
            # Load tokenizer
            model_name = "distilbert-base-uncased"  # Default, should be configurable
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model state
            state_dict = torch.load(self.model_file, map_location=self.device)
            
            # Initialize model architecture
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.intent_labels),
                state_dict=state_dict
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("PyTorch model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model from file: {e}")
            raise
    
    def _initialize_new_model(self):
        """Initialize a new model for training."""
        try:
            model_name = "distilbert-base-uncased"  # Lightweight and effective
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.intent_labels)
            )
            
            self.model.to(self.device)
            logger.info("New PyTorch model initialized for training")
            
        except Exception as e:
            logger.error(f"Failed to initialize new model: {e}")
            raise
    
    def _initialize_fallback_classifier(self):
        """Initialize rule-based fallback classifier."""
        self.model = "fallback_classifier"
        self.tokenizer = "rule_based"
        
        # Define classification rules
        self.classification_rules = {
            "create_reminder": {
                "keywords": ["remind", "reminder", "alert", "alarm", "schedule", "appointment", "meeting"],
                "patterns": [r'\b(remind|reminder)\b', r'\bset\s+.*reminder\b', r'\bappointment\b'],
                "weight": 1.0
            },
            "create_note": {
                "keywords": ["note", "write", "jot", "save", "record", "list", "shopping", "buy"],
                "patterns": [r'\bnote\b', r'\bwrite.*down\b', r'\bshopping.*list\b', r'\bbuy.*groceries\b'],
                "weight": 1.0
            },
            "create_ledger": {
                "keywords": ["owe", "owes", "borrowed", "lent", "paid", "money", "dollar", "expense"],
                "patterns": [r'\b(owe|owes|owed)\b', r'\$\d+', r'\b\d+.*dollars?\b'],
                "weight": 1.0
            },
            "chit_chat": {
                "keywords": ["hello", "hi", "how", "what", "thank", "please", "help"],
                "patterns": [r'\b(hello|hi|hey)\b', r'\bhow.*you\b', r'\bthank.*you\b'],
                "weight": 0.8
            }
        }
        
        logger.info("Fallback rule-based classifier initialized")
    
    def create_training_data(self) -> Tuple[List[str], List[int]]:
        """
        Create comprehensive training data for intent classification.
        This would normally load from your dataset files.
        """
        training_examples = {
            "create_reminder": [
                "Remind me to call John at 5 PM",
                "Set a reminder for my doctor appointment tomorrow",
                "Don't forget to take medicine at 8 AM",
                "Schedule a meeting with the team",
                "Alert me about the deadline next week",
                "Remind me to pick up groceries after work",
                "Set alarm for workout session",
                "Remember to call mom tonight",
                "Appointment with dentist at 2 PM tomorrow",
                "Remind me to submit the report by Friday"
            ],
            "create_note": [
                "Note to buy milk and bread",
                "Write down the meeting agenda",
                "Save this idea for the project",
                "Create a shopping list",
                "Jot down the phone number",
                "Record the discussion points",
                "Make a note about the client feedback",
                "Add to my grocery list",
                "Note about today's meeting",
                "Write down the recipe ingredients"
            ],
            "create_ledger": [
                "John owes me 50 dollars",
                "I borrowed 100 dollars from Sarah",
                "Mark paid me back 25 dollars",
                "Expense for lunch was 15 dollars",
                "Sarah owes me money for dinner",
                "I lent 200 dollars to my friend",
                "Paid 80 dollars for gas",
                "Mike owes me 30 bucks",
                "Cost of groceries was 120 dollars",
                "Borrowed 75 dollars for taxi"
            ],
            "chit_chat": [
                "Hello, how are you?",
                "What can you help me with?",
                "Thank you for your assistance",
                "Good morning!",
                "How's the weather today?",
                "Can you help me?",
                "What time is it?",
                "Please assist me",
                "Hi there!",
                "How does this work?"
            ]
        }
        
        texts = []
        labels = []
        
        for intent, examples in training_examples.items():
            for example in examples:
                texts.append(example)
                labels.append(self.label_to_id[intent])
        
        # Add multi-intent examples
        multi_intent_examples = [
            "Remind me to call John at 5 PM and note to buy groceries",
            "Set reminder for meeting and John owes me 50 dollars",
            "Create a shopping list and remind me about the deadline",
            "Note about project idea and schedule appointment tomorrow",
            "I borrowed 100 dollars from Sarah and remind me to pay back",
        ]
        
        for example in multi_intent_examples:
            # For training, we'll use the primary intent
            if "remind" in example.lower():
                labels.append(self.label_to_id["create_reminder"])
            elif "note" in example.lower():
                labels.append(self.label_to_id["create_note"])
            elif "owe" in example.lower() or "dollar" in example.lower():
                labels.append(self.label_to_id["create_ledger"])
            else:
                labels.append(self.label_to_id["chit_chat"])
            texts.append(example)
        
        return texts, labels
    
    async def train_model(self, epochs: int = 3, learning_rate: float = 2e-5, batch_size: int = 16):
        """
        Train the intent classification model.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
        """
        if not PYTORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.error("PyTorch/Transformers not available for training")
            return False
        
        try:
            logger.info("Starting model training...")
            
            # Get training data
            texts, labels = self.create_training_data()
            logger.info(f"Training with {len(texts)} examples")
            
            # Split data
            if SKLEARN_AVAILABLE:
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    texts, labels, test_size=0.2, random_state=42
                )
            else:
                # Simple split without sklearn
                split_idx = int(0.8 * len(texts))
                train_texts, val_texts = texts[:split_idx], texts[split_idx:]
                train_labels, val_labels = labels[:split_idx], labels[split_idx:]
            
            # Create datasets
            train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./models/training_output',
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                learning_rate=learning_rate,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )
            
            # Train the model
            logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            self.save_model()
            logger.info("Model training completed successfully")
            
            # Evaluate
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def save_model(self):
        """Save the trained model."""
        try:
            # Save model state dict
            torch.save(self.model.state_dict(), self.model_file)
            
            # Save tokenizer
            if self.tokenizer and hasattr(self.tokenizer, 'save_pretrained'):
                tokenizer_path = self.model_path / "tokenizer"
                tokenizer_path.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save_pretrained(tokenizer_path)
            
            # Save label mappings
            mappings = {
                'intent_labels': self.intent_labels,
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }
            
            with open(self.model_path / "label_mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Model saved to {self.model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    async def classify_intent(self, text: str, multi_intent: bool = True) -> Dict[str, Any]:
        """
        Classify intent(s) in the given text.
        
        Args:
            text: Input text to classify
            multi_intent: Whether to detect multiple intents
            
        Returns:
            Dictionary with classification results
        """
        try:
            if multi_intent:
                return await self._classify_multi_intent(text)
            else:
                return await self._classify_single_intent(text)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._create_error_response(text, str(e))
    
    async def _classify_single_intent(self, text: str) -> Dict[str, Any]:
        """Classify a single intent."""
        try:
            if PYTORCH_AVAILABLE and self.model and self.model != "fallback_classifier":
                return await self._pytorch_classify(text)
            else:
                return await self._fallback_classify(text)
        except Exception as e:
            logger.error(f"Single intent classification failed: {e}")
            return self._create_error_response(text, str(e))
    
    async def _pytorch_classify(self, text: str) -> Dict[str, Any]:
        """Classify using PyTorch model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            intent = self.id_to_label[predicted_class]
            
            # Extract entities
            entities = self._extract_entities(text, intent)
            
            return {
                "intent": intent,
                "confidence": float(confidence),
                "entities": entities,
                "original_text": text,
                "model_used": "pytorch",
                "all_predictions": {
                    self.id_to_label[i]: float(predictions[0][i].item()) 
                    for i in range(len(self.intent_labels))
                }
            }
            
        except Exception as e:
            logger.error(f"PyTorch classification failed: {e}")
            return await self._fallback_classify(text)
    
    async def _fallback_classify(self, text: str) -> Dict[str, Any]:
        """Classify using rule-based fallback."""
        try:
            text_lower = text.lower()
            scores = {}
            
            for intent, rules in self.classification_rules.items():
                score = 0.0
                
                # Keyword matching
                for keyword in rules["keywords"]:
                    if keyword in text_lower:
                        score += rules["weight"]
                
                # Pattern matching
                for pattern in rules["patterns"]:
                    if re.search(pattern, text_lower):
                        score += rules["weight"] * 1.5
                
                scores[intent] = score
            
            # Get best intent
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] / 3.0, 1.0)  # Normalize confidence
            
            # Extract entities
            entities = self._extract_entities(text, best_intent)
            
            return {
                "intent": best_intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": text,
                "model_used": "fallback",
                "all_predictions": scores
            }
            
        except Exception as e:
            logger.error(f"Fallback classification failed: {e}")
            return self._create_error_response(text, str(e))
    
    async def _classify_multi_intent(self, text: str) -> Dict[str, Any]:
        """Detect and classify multiple intents in text."""
        try:
            # First, check if text contains multiple intents
            segments = self._segment_text_for_multi_intent(text)
            
            if len(segments) <= 1:
                # Single intent
                result = await self._classify_single_intent(text)
                return {
                    "type": "single_intent",
                    "intent": result["intent"],
                    "confidence": result["confidence"],
                    "entities": result["entities"],
                    "original_text": text,
                    "segments": [text],
                    "results": [result]
                }
            
            # Multiple intents detected
            results = []
            for i, segment in enumerate(segments):
                segment_result = await self._classify_single_intent(segment.strip())
                segment_result["segment_index"] = i
                segment_result["segment_text"] = segment.strip()
                results.append(segment_result)
            
            # Determine overall confidence
            overall_confidence = sum(r["confidence"] for r in results) / len(results)
            
            return {
                "type": "multi_intent",
                "intents": [r["intent"] for r in results],
                "overall_confidence": overall_confidence,
                "original_text": text,
                "segments": segments,
                "results": results,
                "num_intents": len(results)
            }
            
        except Exception as e:
            logger.error(f"Multi-intent classification failed: {e}")
            return self._create_error_response(text, str(e))
    
    def _segment_text_for_multi_intent(self, text: str) -> List[str]:
        """Segment text into potential multiple intents."""
        segments = [text]  # Start with original text
        
        for pattern in self.multi_intent_patterns:
            new_segments = []
            for segment in segments:
                # Split by pattern but keep the separator
                parts = re.split(f'({pattern})', segment, flags=re.IGNORECASE)
                
                if len(parts) > 1:
                    current_segment = ""
                    for i, part in enumerate(parts):
                        if re.match(pattern, part, re.IGNORECASE):
                            # This is a separator, start new segment
                            if current_segment.strip():
                                new_segments.append(current_segment.strip())
                            current_segment = part
                        else:
                            current_segment += part
                    
                    if current_segment.strip():
                        new_segments.append(current_segment.strip())
                else:
                    new_segments.append(segment)
            
            segments = new_segments
        
        # Clean up segments
        cleaned_segments = []
        for segment in segments:
            segment = segment.strip()
            if segment and len(segment) > 3:  # Minimum meaningful length
                cleaned_segments.append(segment)
        
        return cleaned_segments if cleaned_segments else [text]
    
    def _extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities based on intent type."""
        entities = {}
        text_lower = text.lower()
        
        if intent == "create_reminder":
            # Extract time
            time_patterns = [
                r'\b(\d{1,2}:\d{2}\s*(?:am|pm))\b',
                r'\b(\d{1,2}\s*(?:am|pm))\b',
                r'\b(tomorrow|today|tonight)\b',
                r'\b(at|on|by)\s+([^,\.]+)',
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities["time"] = match.group(1) if len(match.groups()) == 1 else match.group(2)
                    break
            
            # Extract reminder title
            title_patterns = [
                r'remind me to (.+?)(?:\s+at|\s+on|\s+tomorrow|\s+today|$)',
                r'reminder (?:for|to) (.+?)(?:\s+at|\s+on|\s+tomorrow|\s+today|$)',
                r'set (?:a )?reminder (?:for|to) (.+?)(?:\s+at|\s+on|\s+tomorrow|\s+today|$)',
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities["title"] = match.group(1).strip()
                    break
        
        elif intent == "create_note":
            # Extract note content
            content_patterns = [
                r'note (?:to|about|that) (.+)',
                r'(?:write|jot|save|record) (?:down )?(.*)',
                r'(?:shopping|grocery) list:? (.*)',
                r'note (.+)',
            ]
            
            for pattern in content_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities["content"] = match.group(1).strip()
                    break
            
            if not entities.get("content"):
                entities["content"] = text
        
        elif intent == "create_ledger":
            # Extract amount
            amount_patterns = [
                r'\$(\d+(?:\.\d{2})?)',
                r'(\d+)\s*dollars?',
                r'(\d+)\s*bucks?',
            ]
            
            for pattern in amount_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities["amount"] = float(match.group(1))
                    break
            
            # Extract person/contact
            person_patterns = [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owes?|owed?|borrowed?|lent)',
                r'(?:owes?|owed?|borrowed?|lent).*?(?:to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                r'(?:I|i)\s+(?:owe|borrowed|lent).*?(?:to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            ]
            
            for pattern in person_patterns:
                match = re.search(pattern, text)
                if match:
                    entities["contact_name"] = match.group(1).strip()
                    break
            
            # Extract transaction type
            if "owe" in text_lower or "debt" in text_lower:
                entities["transaction_type"] = "debt"
            elif "lent" in text_lower or "borrowed" in text_lower:
                entities["transaction_type"] = "loan"
            elif "paid" in text_lower or "expense" in text_lower:
                entities["transaction_type"] = "expense"
            else:
                entities["transaction_type"] = "general"
        
        return entities
    
    def _create_error_response(self, text: str, error: str) -> Dict[str, Any]:
        """Create error response for classification failures."""
        return {
            "intent": "chit_chat",  # Default fallback
            "confidence": 0.1,
            "entities": {},
            "original_text": text,
            "error": error,
            "model_used": "error_fallback"
        }
    
    async def process_for_database(self, classification_result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Process classification results for database storage.
        
        Args:
            classification_result: Result from classify_intent
            user_id: User ID for database records
            
        Returns:
            Processed data ready for database insertion
        """
        try:
            if classification_result.get("type") == "multi_intent":
                return await self._process_multi_intent_for_db(classification_result, user_id)
            else:
                return await self._process_single_intent_for_db(classification_result, user_id)
        except Exception as e:
            logger.error(f"Database processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_single_intent_for_db(self, result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Process single intent for database storage."""
        intent = result["intent"]
        entities = result["entities"]
        
        db_data = {
            "user_id": user_id,
            "original_text": result["original_text"],
            "intent": intent,
            "confidence": result["confidence"],
            "created_at": datetime.now()
        }
        
        if intent == "create_reminder":
            db_data.update({
                "table": "reminders",
                "data": {
                    "title": entities.get("title", result["original_text"]),
                    "time": entities.get("time"),
                    "user_id": user_id,
                    "status": "active"
                }
            })
        
        elif intent == "create_note":
            db_data.update({
                "table": "notes",
                "data": {
                    "content": entities.get("content", result["original_text"]),
                    "user_id": user_id,
                    "category": "general"
                }
            })
        
        elif intent == "create_ledger":
            db_data.update({
                "table": "ledger_entries",
                "data": {
                    "amount": entities.get("amount", 0.0),
                    "contact_name": entities.get("contact_name"),
                    "transaction_type": entities.get("transaction_type", "general"),
                    "description": result["original_text"],
                    "user_id": user_id
                }
            })
        
        elif intent == "chit_chat":
            db_data.update({
                "table": "history_logs",
                "data": {
                    "message": result["original_text"],
                    "response_type": "chit_chat",
                    "user_id": user_id
                }
            })
        
        return {"success": True, "db_data": db_data}
    
    async def _process_multi_intent_for_db(self, result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Process multiple intents for database storage."""
        db_operations = []
        
        for intent_result in result["results"]:
            single_result = await self._process_single_intent_for_db(intent_result, user_id)
            if single_result["success"]:
                db_operations.append(single_result["db_data"])
        
        return {
            "success": True,
            "type": "multi_intent",
            "db_operations": db_operations,
            "total_operations": len(db_operations)
        }
    
    def is_ready(self) -> bool:
        """Check if the service is ready for classification."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "pytorch_available": PYTORCH_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "device": self.device,
            "model_type": type(self.model).__name__ if self.model else None,
            "intent_labels": self.intent_labels,
            "model_ready": self.is_ready()
        }

# Export the service for use in other modules
__all__ = ["PyTorchIntentService"] 