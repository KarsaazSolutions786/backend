# Multi-Intent Processing Feature

## Overview

The Eindr backend now supports **multi-intent processing**, allowing users to express multiple commands in a single utterance. The system can detect, classify, and process multiple intents from one voice input or text, saving each intent to the appropriate database table with independent transactions.

## Features

### ✨ **Multi-Intent Detection**
- Automatically detects multiple intents in complex utterances
- Segments text using linguistic patterns and separators
- Maintains high accuracy for both single and multi-intent scenarios

### 🎯 **Independent Processing**
- Each intent is processed in its own database transaction
- Partial success handling - if one intent fails, others can still succeed
- Detailed results tracking for each intent

### 🔄 **Backward Compatibility**
- Existing single-intent processing continues to work unchanged
- API endpoints accept both single and multi-intent formats
- Graceful fallback for unsupported scenarios

## Supported Intent Combinations

The system can handle any combination of these intent types:

| Intent Type | Database Table | Example |
|-------------|----------------|---------|
| `create_reminder` | `reminders` | "Remind me to call John at 5 PM" |
| `create_note` | `notes` | "Note that I need to buy milk" |
| `create_ledger` | `ledger_entries` | "Sarah owes me $50" |
| `add_expense` | `ledger_entries` | "Track expense: $25 for lunch" |
| `chit_chat` | `history_logs` | "Hello, how are you?" |
| `general_query` | `history_logs` | "What can you help me with?" |

## Example Multi-Intent Utterances

### **Reminder + Note**
```
"Set a reminder for 1 a.m. to sleep and set a note to buy chocolate"
```
**Result**: Creates 1 reminder + 1 note

### **Multiple Reminders**
```
"Remind me to call John at 5 PM and also set a reminder to pick up groceries"
```
**Result**: Creates 2 separate reminders

### **Complex Multi-Intent**
```
"Sarah owes me $50 and note that I need to buy milk plus remind me to call mom"
```
**Result**: Creates 1 ledger entry + 1 note + 1 reminder

## API Integration

### Voice-to-Database Pipeline

The enhanced `/transcribe-and-respond` endpoint automatically handles multi-intent processing:

```bash
curl -X POST "http://localhost:8000/api/v1/stt/transcribe-and-respond" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -F "audio_file=@multi_intent_audio.wav"
```

**Response for Multi-Intent:**
```json
{
  "success": true,
  "pipeline_completed": true,
  "processing_steps": {
    "audio_validation": true,
    "transcription": true,
    "intent_classification": true,
    "database_processing": true
  },
  "transcription": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate",
  "intent_result": {
    "intents": [
      {
        "type": "create_reminder",
        "confidence": 0.93,
        "entities": {"time": "01:00", "title": "sleep"},
        "text_segment": "Set a reminder for 1 a.m. to sleep"
      },
      {
        "type": "create_note",
        "confidence": 0.88,
        "entities": {"content": "buy chocolate"},
        "text_segment": "set a note to buy chocolate"
      }
    ],
    "original_text": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate"
  },
  "processing_result": {
    "success": true,
    "message": "Processed 2 intents successfully",
    "results": [
      {
        "intent": "create_reminder",
        "success": true,
        "data": {
          "reminder_id": "uuid-123",
          "title": "Sleep",
          "time": "2024-01-15T01:00:00Z"
        },
        "position": 1
      },
      {
        "intent": "create_note",
        "success": true,
        "data": {
          "note_id": "uuid-456",
          "content": "buy chocolate"
        },
        "position": 2
      }
    ],
    "total_intents": 2,
    "successful_intents": 2,
    "original_text": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate"
  },
  "ai_response": {
    "text": "Excellent! I've completed both tasks: ✓ Created reminder: 'Sleep' for 01:00 and ✓ Saved note: 'buy chocolate'. Anything else I can do for you?"
  }
}
```

### Direct Intent Processing

Test multi-intent processing directly with the `/intent-processor/test-multi-intent` endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/intent-processor/test-multi-intent" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate",
    "multi_intent": true
  }'
```

### Manual Intent Processing

Process pre-classified intents using `/intent-processor/process`:

```bash
curl -X POST "http://localhost:8000/api/v1/intent-processor/process" \
  -H "Authorization: Bearer $FIREBASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "intent_data": {
      "intents": [
        {
          "type": "create_reminder",
          "confidence": 0.93,
          "entities": {"time": "01:00", "title": "sleep"},
          "text_segment": "Set a reminder for 1 a.m. to sleep"
        },
        {
          "type": "create_note",
          "confidence": 0.88,
          "entities": {"content": "buy chocolate"},
          "text_segment": "set a note to buy chocolate"
        }
      ],
      "original_text": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate"
    }
  }'
```

## Technical Implementation

### Intent Classification

The `IntentService` now supports multi-intent detection:

```python
# Multi-intent classification (default)
intent_result = await intent_service.classify_intent(text, multi_intent=True)

# Single-intent classification (backward compatibility)
intent_result = await intent_service.classify_intent(text, multi_intent=False)
```

**Multi-Intent Response Format:**
```python
{
    "intents": [
        {
            "type": "create_reminder",
            "confidence": 0.93,
            "entities": {"time": "01:00", "title": "sleep"},
            "text_segment": "Set a reminder for 1 a.m. to sleep"
        },
        {
            "type": "create_note",
            "confidence": 0.88,
            "entities": {"content": "buy chocolate"},
            "text_segment": "set a note to buy chocolate"
        }
    ],
    "original_text": "set a reminder for 1 a.m. to sleep and set a note to buy chocolate"
}
```

### Text Segmentation

The system uses sophisticated pattern matching to segment multi-intent utterances:

**Separator Patterns:**
- `"and set"`, `"and create"`, `"and add"`
- `"also set"`, `"also create"`
- `"then set"`, `"then create"`
- `"plus set"`, `"plus create"`

**Example Segmentation:**
```
Input: "Set a reminder for 1 a.m. to sleep and set a note to buy chocolate"

Segments:
1. "Set a reminder for 1 a.m. to sleep"
2. "set a note to buy chocolate"
```

### Database Processing

The `IntentProcessorService` processes each intent independently:

```python
# Handler registry for easy extensibility
self.intent_handlers = {
    "create_reminder": self._process_reminder,
    "create_note": self._process_note,
    "create_ledger": self._process_ledger,
    "add_expense": self._process_ledger,
    "chit_chat": self._process_chat,
    "general_query": self._process_chat,
}
```

**Independent Transactions:**
- Each intent gets its own database session
- Failed intents don't affect successful ones
- Detailed error reporting per intent

## Testing

### Manual Testing

Run the comprehensive test script:

```bash
python test_multi_intent.py
```

**Test Cases Included:**
- Reminder + Note combinations
- Multiple reminders
- Ledger + Note + Reminder
- Single intent (baseline)
- Complex multi-intent scenarios

### API Testing

Use the test endpoint for quick validation:

```bash
# Test multi-intent processing
curl -X POST "http://localhost:8000/api/v1/intent-processor/test-multi-intent" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "remind me to call John and note to buy milk", "multi_intent": true}'
```

### Unit Testing Examples

```python
async def test_multi_intent_classification():
    intent_service = IntentService()
    
    result = await intent_service.classify_intent(
        "Set a reminder for 5 PM and create a note about groceries",
        multi_intent=True
    )
    
    assert "intents" in result
    assert len(result["intents"]) == 2
    assert result["intents"][0]["type"] == "create_reminder"
    assert result["intents"][1]["type"] == "create_note"

async def test_independent_transactions():
    processor = IntentProcessorService()
    
    # Test with one valid and one invalid intent
    intent_data = {
        "intents": [
            {"type": "create_reminder", "confidence": 0.9, "entities": {}},
            {"type": "invalid_intent", "confidence": 0.8, "entities": {}}
        ],
        "original_text": "test utterance"
    }
    
    result = await processor.process_multi_intent(intent_data, "test_user")
    
    assert result["total_intents"] == 2
    assert result["successful_intents"] == 1  # Only valid intent succeeds
    assert result["results"][0]["success"] == True
    assert result["results"][1]["success"] == False
```

## Error Handling

### Partial Success

When some intents succeed and others fail:

```json
{
  "success": false,
  "message": "Processed 2 intents (with some failures)",
  "results": [
    {
      "intent": "create_reminder",
      "success": true,
      "data": {"reminder_id": "uuid-123"}
    },
    {
      "intent": "invalid_intent",
      "success": false,
      "error": "Unknown intent type"
    }
  ],
  "total_intents": 2,
  "successful_intents": 1
}
```

### Complete Failure

When all intents fail:

```json
{
  "success": false,
  "message": "Processed 2 intents (with some failures)",
  "results": [
    {
      "intent": "create_reminder",
      "success": false,
      "error": "Failed to extract time information"
    },
    {
      "intent": "create_note",
      "success": false,
      "error": "Empty content not allowed"
    }
  ],
  "total_intents": 2,
  "successful_intents": 0
}
```

## Performance Considerations

### Database Optimization

- **Independent Sessions**: Each intent uses its own database session
- **Transaction Isolation**: Failed intents don't rollback successful ones
- **Batch Validation**: User validation performed once for all intents

### Memory Usage

- **Streaming Processing**: Intents processed sequentially, not held in memory
- **Cleanup**: Database sessions properly closed after each intent
- **Error Recovery**: Failed processing doesn't leak resources

### Scalability

- **Stateless Design**: No state maintained between requests
- **Parallel Potential**: Intent processing can be parallelized in future versions
- **Caching**: Intent patterns and models can be cached for performance

## Future Enhancements

### Intent Relationship Handling
- **Cross-Intent References**: "Remind me about the meeting I just noted"
- **Conditional Processing**: "If Sarah pays me back, create a note about it"
- **Intent Priorities**: Processing order based on importance

### Advanced Segmentation
- **NLP-Based Segmentation**: Using transformer models for better accuracy
- **Context Awareness**: Understanding pronoun references across segments
- **Ambiguity Resolution**: Handling unclear intent boundaries

### Performance Optimizations
- **Parallel Processing**: Concurrent database operations for independent intents
- **Batch Operations**: Optimized database queries for multiple intents
- **Caching Layer**: Intent classification caching for common patterns

## Troubleshooting

### Common Issues

**Intent Not Detected:**
- Check if text contains supported separator patterns
- Verify intent keywords are present in each segment
- Test with simpler utterances first

**Database Processing Fails:**
- Check user authentication and permissions
- Verify database connectivity
- Review entity extraction for required fields

**Partial Success:**
- Review individual intent results in the response
- Check entity extraction for failed intents
- Validate input format for each intent type

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('services.intent_service').setLevel(logging.DEBUG)
logging.getLogger('services.intent_processor_service').setLevel(logging.DEBUG)
```

## Conclusion

The multi-intent processing feature significantly enhances the Eindr backend's capability to handle complex user utterances. With independent transaction processing, comprehensive error handling, and backward compatibility, the system can now efficiently process multiple commands in a single interaction while maintaining reliability and performance.

The implementation follows best practices for:
- **Modularity**: Clean separation between classification and processing
- **Extensibility**: Easy addition of new intent types and handlers
- **Reliability**: Independent transactions and comprehensive error handling
- **Performance**: Efficient processing with proper resource management

This feature opens up new possibilities for natural, conversational interactions with the Eindr system, making it more intuitive and powerful for users. 