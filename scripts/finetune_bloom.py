#!/usr/bin/env python3
"""
Fine-tune Bloom-560M-8bit for general chat using LoRA
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BloomChatTrainer:
    """Trainer for fine-tuning Bloom model with LoRA for general chat."""
    
    def __init__(
        self,
        base_model_path: str = "Models/bloom-560m-8bit",
        output_dir: str = "Models/bloom-560m-8bit-finetuned",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query_key_value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Initialize model and tokenizer
        self.setup_model_and_tokenizer()
    
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Loading base model from {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32,
                device_map=None
            )

            # Apply LoRA
            logger.info("Applying LoRA adapter")
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def prepare_training_data(self, examples: List[Dict[str, Any]]) -> Dataset:
        """Prepare training data in the correct format."""
        formatted_data = []
        
        for example in examples:
            # Format the conversation
            conversation = self._format_conversation(example)
            
            # Tokenize the conversation
            tokenized = self.tokenizer(
                conversation,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Add labels for the response part
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            
            formatted_example = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            }
            
            formatted_data.append(formatted_example)
        
        return Dataset.from_list(formatted_data)
    
    def _format_conversation(self, example: Dict[str, Any]) -> str:
        """Format a single conversation example.

        Supports both the legacy {{"text", "response"}} format and the new
        {{"system", "user", "assistant"}} format. This allows a mixed dataset
        during the transition period.
        """
        # Legacy keys
        if "text" in example and "response" in example:
            system_msg = "You are a friendly AI assistant who enjoys casual conversations and helping people."
            user_msg = example["text"]
            assistant_msg = example["response"]
        # New recommended keys
        else:
            system_msg = example.get("system", "You are a helpful assistant.")
            user_msg = example.get("user", "")
            assistant_msg = example.get("assistant", "")

        return f"SYSTEM: {system_msg}\nUSER: {user_msg}\nASSISTANT: {assistant_msg}"
    
    def train(
        self,
        train_dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_steps: int = 1000
    ):
        """Train the model."""
        try:
            logger.info("Starting training")
            logger.info(f"Training on {len(train_dataset)} examples")
            logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                max_steps=max_steps,
                logging_steps=10,
                save_steps=200,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=f"{self.output_dir}/logs",
                use_cpu=True,  # Use CPU instead of GPU
                remove_unused_columns=False,
                report_to="none",  # Disable wandb tracking
                save_total_limit=2  # Keep only the last 2 checkpoints
            )
        
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
        
            # Train
            logger.info("Training started…")
            train_result = trainer.train()
        
            # Log training results
            logger.info("Training completed!")
            logger.info(f"Total training time: {train_result.metrics['train_runtime']:.2f} seconds")
            logger.info(f"Training loss: {train_result.metrics['train_loss']:.4f}")
        
            # Save the model
            logger.info(f"Saving model to {self.output_dir}")
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def test_model(self, test_examples: List[Dict[str, Any]]):
        """Test the fine-tuned model."""
        logger.info("Testing fine-tuned model")
        
        for example in test_examples[:5]:  # Test first 5 examples
            prompt = self._format_conversation(example)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            input_display = example.get("text") or example.get("user", "<unknown>")
            logger.info(f"\nInput: {input_display}\nOutput: {response}\n")

def main():
    """Main training function."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Bloom for general chat")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--data-file", type=str, default="data/chat_data.jsonl")
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BloomChatTrainer()
    
    # Load training data
    logger.info(f"Loading training data from {args.data_file}")
    with open(args.data_file) as f:
        examples = [json.loads(line) for line in f]
    
    # Prepare dataset
    dataset = trainer.prepare_training_data(examples)
    
    # Train
    trainer.train(
        train_dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps
    )
    
    # Test
    logger.info("Testing the fine-tuned model")
    trainer.test_model(examples[-5:])  # Test on last 5 examples

if __name__ == "__main__":
    main() 