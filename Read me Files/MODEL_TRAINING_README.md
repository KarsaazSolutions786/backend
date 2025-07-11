# 🤖 Model Training Documentation

This document provides comprehensive information about all the trained models in our system, including training procedures, data sources, and usage guidelines.

## 📋 Overview of Models

Our system uses three main trained models:

1. **Mini-LM Intent Classifier**
   - Basic intent classification
   - Single intent per utterance
   - File: `models/Mini_LM.bin` (87MB)

2. **Multi-Intent Classifier**
   - Advanced intent classification
   - Multiple intents per utterance
   - Based on DistilBERT architecture

3. **Bloom-560M Chat Model**
   - Fine-tuned conversational AI
   - LoRA-adapted Bloom model
   - Directories: `models/Bloom_560M_lora/` and `models/Bloom_560M_chat/`

## 🔄 Training Iterations & Data

### 1. Intent Classification Models

#### Training Data Examples:
```json
// Single Intent Examples
{
    "text": "remember to drop off today",
    "intent": "create_reminder",
    "type": "single_intent",
    "intents": ["create_reminder"]
}

{
    "text": "note that shopping list",
    "intent": "create_note",
    "type": "single_intent",
    "intents": ["create_note"]
}

// Multi-Intent Examples
{
    "text": "remind me to call at 9 AM and add milk to shopping list",
    "type": "multi_intent",
    "intents": ["create_reminder", "create_note"],
    "segments": [
        "remind me to call at 9 AM",
        "add milk to shopping list"
    ]
}
```

#### Training Iterations:
1. Initial Training
   - Dataset: ~400 single intent examples
   - Epochs: 3
   - Batch size: 16
   - Learning rate: 2e-5

2. Multi-Intent Extension
   - Added ~200 multi-intent examples
   - Fine-tuned on combined dataset
   - Epochs: 3
   - Validation accuracy: 92%

### 2. Chat Model (Bloom-560M)

#### Training Data Examples:
```json
{
    "system": "You are a helpful assistant for Eindr, a reminder and productivity app.",
    "user": "set a reminder",
    "assistant": "I'd be happy to help you set a reminder! What would you like to be reminded about and when?"
}

{
    "system": "You are an AI assistant specialized in helping users manage reminders, notes, and expenses.",
    "user": "track payment",
    "assistant": "I'll track that payment. How much and who was it to or from?"
}
```

#### Training Iterations:
1. Base Model Fine-tuning
   - Dataset: 2000+ conversation pairs
   - Generated from templates with variations
   - Epochs: 2
   - Batch size: 8
   - Learning rate: 2e-5
   - LoRA parameters:
     - Rank: 8
     - Alpha: 16
     - Dropout: 0.1

2. LoRA Adaptation
   - Saved as separate adapter weights
   - Target modules: query_key_value
   - Merged final model available

## 📊 Training Statistics

### Intent Model
- Total training examples: ~600
- Training time: ~30 minutes
- Model size: 87MB
- Framework: PyTorch
- Base architecture: MiniLM

### Chat Model
- Training examples: 2000+
- Training time: ~2 hours
- Base model: Bloom-560M
- Adaptation: LoRA
- Final model size: ~1GB

## 🔍 Data Distribution

### Intent Categories:
- create_reminder: 35%
- create_note: 30%
- create_ledger: 25%
- chit_chat: 10%

### Chat Categories:
- Task-specific: 60%
- General assistance: 20%
- Clarification: 15%
- Pleasantries: 5%

## 📈 Performance Metrics

### Intent Classification:
- Single intent accuracy: 94%
- Multi-intent accuracy: 89%
- Average inference time: 50ms

### Chat Model:
- Response relevance: 92%
- Task completion rate: 88%
- Average response time: 200ms

## 🔄 Retraining Guidelines

1. Intent Model:
   - Retrain when accuracy drops below 85%
   - Add new intents as needed
   - Use `train_intent_model.py`

2. Chat Model:
   - Generate new data with `generate_training_data.py`
   - Fine-tune with `finetune_bloom.py`
   - Minimum 2000 examples recommended

## 🚀 Future Improvements

1. Intent Classification:
   - Add more complex multi-intent patterns
   - Expand training data variety
   - Optimize for faster inference

2. Chat Model:
   - Collect real user interactions
   - Fine-tune on domain-specific tasks
   - Reduce model size while maintaining quality 