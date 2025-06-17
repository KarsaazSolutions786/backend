# Fine-tuning Bloom-560M for General Chat

This guide explains how to fine-tune the Bloom-560M model for better general chat capabilities using LoRA (Low-Rank Adaptation).

## Prerequisites

- Python 3.11
- Base Bloom-560M-8bit model in `Models/bloom-560m-8bit/`
- At least 8GB RAM for CPU training

## Setup

1. Create a virtual environment:
   ```bash
   python3.11 -m venv venv-pytorch
   source venv-pytorch/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements_finetune.txt
   ```

## Training Data

The training data is stored in `data/chat_data.jsonl` in the following format:
```json
{
    "text": "User message",
    "response": "Assistant response",
    "type": "message_type"
}
```

Message types include:
- greeting
- casual_question
- gratitude
- compliment
- farewell
- empathy
- introduction

## Fine-tuning

Run the fine-tuning script:
```bash
python scripts/finetune_bloom.py \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --max-steps 500
```

Parameters:
- `epochs`: Number of training epochs (default: 3)
- `batch-size`: Batch size for training (default: 8, use 4 for CPU)
- `learning-rate`: Learning rate (default: 2e-5)
- `max-steps`: Maximum number of training steps (default: 1000)
- `data-file`: Path to training data (default: data/chat_data.jsonl)

## LoRA Configuration

The model uses the following LoRA settings:
- Rank (r): 8
- Alpha (α): 16
- Target modules: ["query_key_value"]
- Dropout: 0.1

## Output

The fine-tuned model will be saved in `Models/bloom-560m-8bit-finetuned/` with:
- Model weights
- Tokenizer files
- Training logs

## Testing

The script automatically tests the fine-tuned model on the last 5 examples from the training data. The results are logged to:
- Console output
- `finetune.log`
- `models/Bloom_560M_finetuned/logs/training.log`

## Integration

After fine-tuning, update the model path in your chat service to use the fine-tuned model:
```python
model_path = "Models/bloom-560m-8bit-finetuned"
```

## Monitoring

Monitor training progress in real-time:
```bash
tail -f finetune.log
```

## Troubleshooting

1. If you get CUDA/GPU errors, the script will automatically fall back to CPU training.
2. If you encounter memory issues, try:
   - Reducing batch size
   - Reducing max_steps
   - Using gradient accumulation (already set to 4)

## Notes

- The training uses CPU by default for compatibility
- Training progress is logged every 10 steps
- Model checkpoints are saved every 200 steps
- Only the last 2 checkpoints are kept to save space 