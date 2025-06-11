#!/usr/bin/env python3
"""
Fine-tune Bloom-560M for chat assistant behavior using PEFT/LoRA.

Usage:
    python scripts/finetune_bloom.py

Requirements:
    - Base model weights at Models/bloom560m.bin
    - Training data at data/chat_pairs.jsonl
    - At least 2000 training examples
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from pathlib import Path
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BloomChatFineTuner:
    """Fine-tune Bloom-560M for chat assistant behavior."""
    
    def __init__(self):
        self.base_model_path = "Models/bloom560m.bin"
        self.training_data_path = "data/chat_pairs.jsonl"
        self.lora_output_path = "Models/Bloom_560M_lora"
        self.merged_output_path = "Models/Bloom_560M_chat"
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_key_value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir="./training_output",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True if torch.cuda.is_available() else False,
        )
    
    def load_base_model(self):
        """Load base Bloom-560M model from binary weights."""
        logger.info("Loading base Bloom-560M model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Load custom weights if available
        if os.path.exists(self.base_model_path):
            logger.info(f"Loading custom weights from {self.base_model_path}")
        try:
                custom_weights = torch.load(self.base_model_path, map_location='cpu')
                self.model.load_state_dict(custom_weights, strict=False)
            logger.info("Custom weights loaded successfully")
        except Exception as e:
                logger.warning(f"Could not load custom weights: {e}, using base model")
        
        logger.info("Base model loaded successfully")
    
    def load_training_data(self) -> Dataset:
        """Load and preprocess training data from JSONL file."""
        logger.info(f"Loading training data from {self.training_data_path}")
        
        if not os.path.exists(self.training_data_path):
            raise FileNotFoundError(f"Training data not found at {self.training_data_path}")
        
        # Load JSONL data
        data = []
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if all(key in item for key in ['system', 'user', 'assistant']):
                        data.append(item)
                except json.JSONDecodeError:
                    continue
        
        if len(data) < 2000:
            logger.warning(f"Only {len(data)} training examples found, recommended minimum is 2000")
        
        logger.info(f"Loaded {len(data)} training examples")
        
        # Format for chat training
        formatted_data = []
        for item in data:
            # Create conversation format
            conversation = f"System: {item['system']}\nUser: {item['user']}\nAssistant: {item['assistant']}"
            formatted_data.append({"text": conversation})
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training."""
        logger.info("Tokenizing training data...")
        
        def tokenize_function(examples):
            # Tokenize with truncation and padding
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info("Data tokenization completed")
        return tokenized_dataset
    
    def setup_lora(self):
        """Apply LoRA configuration to the model."""
        logger.info("Setting up LoRA configuration...")
        
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup completed")
    
    def train(self, dataset: Dataset):
        """Train the model with LoRA."""
        logger.info("Starting fine-tuning...")
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        trainer.train()
        
        logger.info("Fine-tuning completed")
        return trainer
    
    def save_lora_weights(self):
        """Save LoRA adapter weights."""
        logger.info(f"Saving LoRA weights to {self.lora_output_path}")
        
        os.makedirs(self.lora_output_path, exist_ok=True)
        self.model.save_pretrained(self.lora_output_path)
        self.tokenizer.save_pretrained(self.lora_output_path)
        
        logger.info("LoRA weights saved successfully")
    
    def merge_and_save(self):
        """Merge LoRA weights with base model and save as fp16 checkpoint."""
        logger.info(f"Merging LoRA weights and saving to {self.merged_output_path}")
        
        # Merge LoRA weights
        merged_model = self.model.merge_and_unload()
        
        # Convert to fp16 for efficiency
        if torch.cuda.is_available():
            merged_model = merged_model.half()
        
        # Save merged model
        os.makedirs(self.merged_output_path, exist_ok=True)
        merged_model.save_pretrained(
            self.merged_output_path,
            torch_dtype=torch.float16,
            safe_serialization=True
        )
        self.tokenizer.save_pretrained(self.merged_output_path)
        
        logger.info("Merged model saved successfully")
    
    def run_fine_tuning(self):
        """Execute the complete fine-tuning pipeline."""
        try:
            # Step 1: Load base model
            self.load_base_model()
            
            # Step 2: Load and preprocess training data
            dataset = self.load_training_data()
            tokenized_dataset = self.tokenize_data(dataset)
            
            # Step 3: Setup LoRA
            self.setup_lora()
            
            # Step 4: Train
            trainer = self.train(tokenized_dataset)
            
            # Step 5: Save LoRA weights
            self.save_lora_weights()
            
            # Step 6: Merge and save final model
            self.merge_and_save()
            
            logger.info("Fine-tuning pipeline completed successfully!")
            logger.info(f"LoRA weights saved to: {self.lora_output_path}")
            logger.info(f"Merged model saved to: {self.merged_output_path}")
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise

def main():
    """Main function to run fine-tuning."""
    print("🚀 Starting Bloom-560M Fine-tuning for Chat Assistant")
    print("=" * 60)
    
    # Check prerequisites
    if not os.path.exists("Models/bloom560m.bin"):
        print("❌ Base model not found at Models/bloom560m.bin")
        return
    
    if not os.path.exists("data/chat_pairs.jsonl"):
        print("❌ Training data not found at data/chat_pairs.jsonl")
        print("Please ensure you have at least 2000 chat examples in JSONL format")
        return
    
    # Create directories
    os.makedirs("Models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Run fine-tuning
    fine_tuner = BloomChatFineTuner()
    fine_tuner.run_fine_tuning()
    
    print("✅ Fine-tuning completed successfully!")
    print("The chat model is now ready for use in services/chat_service.py")

if __name__ == "__main__":
    main() 