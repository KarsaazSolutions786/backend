# Multi-Intent Model Training Guide

This guide will help you train a proper machine learning model that can generalize to new words and scenarios, moving away from hardcoded patterns.

## 🎯 Why Train Your Own Model?

**Problems with Hardcoded Patterns:**
- ❌ Can't handle new words or names (e.g., "Zara owes me $50")
- ❌ Breaks with slight variations in phrasing
- ❌ Requires manual updates for every new scenario
- ❌ Poor multi-intent detection for complex sentences

**Benefits of Trained Model:**
- ✅ **Generalizes to new words** - handles any name, place, or item
- ✅ **Understands context** - learns patterns rather than exact matches
- ✅ **Better multi-intent detection** - trained on complex examples
- ✅ **Self-improving** - can be retrained with new data
- ✅ **Robust to variations** - handles different phrasings naturally

## 🚀 Quick Start

### 1. Install Training Dependencies

```bash
# Install training requirements
pip install -r requirements_training.txt

# Or install manually:
pip install torch transformers scikit-learn pandas numpy tqdm datasets accelerate
```

### 2. Generate Training Data (Optional)

```bash
# Generate comprehensive training dataset
python train_multi_intent_model.py --generate-only --data-size 2000
```

This creates `data/multi_intent_training_data.csv` with:
- **Single intent examples**: Individual reminders, notes, ledgers, chat
- **Multi-intent examples**: Complex sentences with multiple intents
- **Data augmentation**: Synonym replacements and variations
- **Real-world scenarios**: Names, amounts, times, locations

### 3. Train the Model

```bash
# Basic training (recommended)
python train_multi_intent_model.py --epochs 5 --batch-size 16

# Advanced training with custom parameters
python train_multi_intent_model.py \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --data-size 3000 \
  --model-name "distilbert-base-uncased"
```

### 4. Test Your Trained Model

```bash
# Test the trained model
python -c "
import asyncio
from services.trained_model_service import TrainedModelService

async def test():
    service = TrainedModelService()
    
    test_cases = [
        'remind me to call Zara at 6 PM',
        'note to buy chocolate and John owes me 75 dollars',
        'set reminder for dentist appointment and note meeting with client at noon',
        'Emily will pay me 200 dollars and remind me to book flight tickets'
    ]
    
    for text in test_cases:
        result = await service.classify_intent(text)
        print(f'Input: {text}')
        print(f'Result: {result}')
        print('-' * 50)

asyncio.run(test())
"
```

## 📊 Training Details

### Dataset Generation

The training system automatically generates diverse examples:

**Single Intent Examples (per intent):**
- **Reminders**: "remind me to {action} at {time}", variations with names, times, etc.
- **Notes**: "note to {content}", "write down {info}", with various content types
- **Ledger**: "{person} owes me ${amount}", financial transactions, names, amounts
- **Chat**: greetings, questions, thanks, help requests

**Multi-Intent Examples:**
- Combined intents with natural connectors ("and", "also", "plus", "then")
- Complex scenarios: "remind me to call John and note that Sarah owes $50"
- Real-world combinations users actually say

**Data Augmentation:**
- Synonym replacement (remind → alert, note → record)
- Politeness variations (please, kindly, could you)
- Natural language variations

### Model Architecture

- **Base Model**: DistilBERT (fast, efficient, good performance)
- **Task**: Multi-class classification with 4 intents
- **Input**: Raw text (up to 256 tokens)
- **Output**: Intent probabilities + confidence scores
- **Multi-Intent**: Intelligent text segmentation + individual classification

### Training Process

1. **Data Split**: 80% training, 20% validation
2. **Optimization**: AdamW optimizer with weight decay
3. **Learning Rate**: 2e-5 (can be adjusted)
4. **Batch Size**: 16 (adjust based on GPU memory)
5. **Epochs**: 5 (more for better performance)
6. **Evaluation**: Accuracy, F1-score, classification report

## 🔧 Advanced Configuration

### Custom Training Data

If you have your own training data:

```python
# Create custom training data CSV with columns:
# text, intent, type
import pandas as pd

data = [
    {"text": "remind me to call Alice", "intent": "create_reminder", "type": "single_intent"},
    {"text": "note that Bob owes me $100", "intent": "create_ledger", "type": "single_intent"},
    # ... more examples
]

df = pd.DataFrame(data)
df.to_csv("data/custom_training_data.csv", index=False)

# Train with custom data
python train_multi_intent_model.py --data-file data/custom_training_data.csv
```

### Model Parameters

```bash
# Experiment with different parameters
python train_multi_intent_model.py \
  --model-name "bert-base-uncased" \    # Larger model for better performance
  --epochs 15 \                         # More training
  --batch-size 8 \                      # Smaller if GPU memory limited
  --learning-rate 1e-5 \                # Lower for fine-tuning
  --data-size 5000                      # More training data
```

### Different Base Models

You can experiment with different base models:

- `distilbert-base-uncased` (default) - Fast, good performance
- `bert-base-uncased` - Better performance, slower
- `roberta-base` - Often best performance
- `albert-base-v2` - Memory efficient

## 🎯 Integration with Your Backend

### Option 1: Update Intent Service (Recommended)

Modify `services/intent_service.py` to use the trained model:

```python
from services.trained_model_service import TrainedModelService

class IntentService:
    def __init__(self):
        self.trained_service = TrainedModelService()
        # ... existing code
    
    async def classify_intent(self, text: str):
        if self.trained_service.is_ready():
            return await self.trained_service.classify_intent(text)
        else:
            # Fallback to existing MiniLM service
            return await self.minilm_service.classify_intent(text)
```

### Option 2: Direct Integration

Use the trained model service directly:

```python
from services.trained_model_service import TrainedModelService

# In your API endpoint
service = TrainedModelService()
result = await service.classify_intent(transcription_text)
```

## 📈 Performance Monitoring

### Evaluation Metrics

After training, you'll see:

```
Evaluation Results:
Accuracy: 0.932
F1 Score: 0.928

Classification Report:
               precision    recall  f1-score   support

create_reminder    0.94      0.91      0.93       150
   create_note     0.89      0.95      0.92       142
  create_ledger    0.96      0.88      0.92       138
      chit_chat    0.92      0.98      0.95       124

    avg / total    0.93      0.93      0.93       554
```

### Real-World Testing

Test with scenarios your users actually encounter:

```python
test_cases = [
    # New names the model hasn't seen
    "Priya owes me 80 dollars for groceries",
    
    # Complex multi-intent
    "set reminder to call the bank tomorrow and note that Alex borrowed my laptop",
    
    # Variations in phrasing
    "could you please remind me about the meeting with Jennifer at 3 PM",
    
    # New locations/items
    "note to buy kimchi from the Korean market and remind me about yoga class"
]
```

## 🔄 Continuous Improvement

### Retraining with New Data

As you collect more user interactions:

1. **Collect real transcriptions** that failed or worked well
2. **Label them correctly** with proper intents
3. **Add to training data** in CSV format
4. **Retrain the model** with updated dataset

```bash
# Retrain with additional data
python train_multi_intent_model.py \
  --data-file data/combined_training_data.csv \
  --epochs 3 \
  --save-path models/Multi_LM_v2.bin
```

### Active Learning

Monitor classification confidence:
- Low confidence predictions (< 0.7) → Review manually
- Add corrected examples to training data
- Retrain periodically

## 🚨 Troubleshooting

### Common Issues

**1. CUDA/GPU Issues:**
```bash
# Force CPU usage if GPU issues
export CUDA_VISIBLE_DEVICES=""
python train_multi_intent_model.py
```

**2. Memory Issues:**
```bash
# Reduce batch size
python train_multi_intent_model.py --batch-size 8
```

**3. Poor Performance:**
- Increase training data size (`--data-size 5000`)
- More epochs (`--epochs 10`)
- Try different base model (`--model-name bert-base-uncased`)

**4. Model Not Loading:**
```bash
# Check if model files exist
ls -la models/Multi_LM*

# Retrain if missing
python train_multi_intent_model.py
```

## 📋 File Structure

After training, you'll have:

```
models/
├── Multi_LM.bin                 # Trained model weights
├── Multi_LM_tokenizer/          # Tokenizer files
│   ├── config.json
│   ├── tokenizer.json
│   └── vocab.txt
└── Mini_LM.bin                  # Original model (backup)

data/
├── multi_intent_training_data.csv    # Generated training data
└── multi_intent_training_data.json   # Same data in JSON format

services/
├── trained_model_service.py          # New trained model service
├── minilm_intent_service.py          # Original service (fallback)
└── intent_service.py                 # Main service (update this)
```

## ⚡ Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements_training.txt

# Generate data only
python train_multi_intent_model.py --generate-only

# Quick training
python train_multi_intent_model.py

# Custom training
python train_multi_intent_model.py --epochs 10 --batch-size 32 --data-size 3000

# Test trained model
python -c "from services.trained_model_service import TrainedModelService; import asyncio; asyncio.run(TrainedModelService().classify_intent('test'))"
```

## 🎉 Expected Results

After training, your model should:

1. **Handle new names**: "Rajesh owes me $45" ✅
2. **Understand variations**: "could you please set up a reminder" ✅
3. **Multi-intent detection**: "call mom and John paid me back" ✅
4. **Context awareness**: "meeting at noon remind me about the project" ✅
5. **Better accuracy**: 90%+ on real-world transcriptions ✅

The model will automatically generalize to new words and scenarios without requiring manual pattern updates! 