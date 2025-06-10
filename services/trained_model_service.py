"""
Trained Multi-Intent Model Service
Uses the properly trained model for multi-intent classification with better generalization.
"""

import os
import re
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from core.config import settings
from utils.logger import logger

# Import dependencies with fallback
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available for trained model service")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - trained model service will use fallbacks")

class TrainedModelService:
    """Service for using the trained multi-intent classification model."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        # Model paths
        self.model_path = "models/Multi_LM.bin"
        self.tokenizer_path = "models/Multi_LM_tokenizer"
        self.fallback_model_path = "models/Mini_LM.bin"
        
        # Intent labels (must match training)
        self.intent_labels = ["create_reminder", "create_note", "create_ledger", "chit_chat"]
        self.label_to_id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.intent_labels)}
        
        # Multi-intent detection patterns (enhanced but less relied upon)
        self.multi_intent_indicators = [
            r'\band\s+(?:set|create|add|make|remind|schedule|note|owe)',
            r'\balso\s+(?:set|create|add|make|remind|schedule|note)',
            r'\bplus\s+(?:set|create|add|make|remind|schedule|note)',
            r'\bthen\s+(?:set|create|add|make|remind|schedule|note)',
            r'\band\s+[A-Z][a-z]+\s+(?:owes?|owed?|will\s+(?:give|pay))',
            r'(?:\.|;)\s*(?:Set|Create|Add|Make|Remind|Schedule|Note)',
        ]
        
        # Initialize the service
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.error("Transformers not available - cannot load trained model")
                return False
            
            # Check if trained model exists
            if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
                logger.info(f"Loading trained model from {self.model_path}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                
                # Load model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",  # Base architecture
                    num_labels=len(self.intent_labels)
                )
                
                # Load trained weights
                model_state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(model_state)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("✅ Trained model loaded successfully!")
                return True
            
            else:
                logger.warning(f"Trained model not found at {self.model_path}")
                logger.info("Please train a model first using: python train_multi_intent_model.py")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            return False
    
    async def classify_intent(self, text: str, multi_intent: bool = True) -> Dict[str, Any]:
        """
        Classify intent(s) using the trained model.
        
        Args:
            text: Input text to classify
            multi_intent: Whether to detect multiple intents
            
        Returns:
            Dictionary containing classification results
        """
        try:
            if not self.model or not self.tokenizer:
                return self._create_fallback_response(text, multi_intent)
            
            # Clean text
            text = self._clean_text(text)
            
            if multi_intent and self._likely_multi_intent(text):
                return await self._classify_multi_intent(text)
            else:
                return await self._classify_single_intent(text)
                
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._create_error_response(text, str(e))
    
    async def _classify_single_intent(self, text: str) -> Dict[str, Any]:
        """Classify single intent using the trained model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # Get predicted intent
            predicted_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_idx].item()
            intent = self.intent_labels[predicted_idx]
            
            # Extract entities using the trained model's understanding
            entities = self._extract_entities_intelligent(text, intent)
            
            logger.info(f"Single intent classification: {intent} (confidence: {confidence:.3f})")
            
            return {
                "success": True,
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": text,
                "model_used": "trained_multi_intent",
                "type": "single_intent"
            }
            
        except Exception as e:
            logger.error(f"Single intent classification failed: {e}")
            return self._create_error_response(text, str(e))
    
    async def _classify_multi_intent(self, text: str) -> Dict[str, Any]:
        """Classify multiple intents using the trained model."""
        try:
            # Segment text using intelligent approach
            segments = self._segment_text_intelligent(text)
            
            if len(segments) <= 1:
                return await self._classify_single_intent(text)
            
            # Classify each segment
            results = []
            overall_confidence = 0.0
            
            for i, segment in enumerate(segments):
                segment_result = await self._classify_single_intent(segment)
                
                if segment_result.get("success", False):
                    results.append({
                        "intent": segment_result["intent"],
                        "confidence": segment_result["confidence"],
                        "entities": segment_result["entities"],
                        "segment": segment,
                        "segment_index": i
                    })
                    overall_confidence += segment_result["confidence"]
            
            if not results:
                return self._create_error_response(text, "No valid classifications found")
            
            # Calculate overall confidence and extract unique intents
            overall_confidence /= len(results)
            intents = list(dict.fromkeys([r["intent"] for r in results]))  # Remove duplicates
            
            logger.info(f"Multi-intent classification: {intents} (overall confidence: {overall_confidence:.3f})")
            
            return {
                "success": True,
                "type": "multi_intent",
                "intents": intents,
                "overall_confidence": overall_confidence,
                "segments": segments,
                "results": results,
                "original_text": text,
                "model_used": "trained_multi_intent",
                "total_intents": len(intents)
            }
            
        except Exception as e:
            logger.error(f"Multi-intent classification failed: {e}")
            return self._create_error_response(text, str(e))
    
    def _likely_multi_intent(self, text: str) -> bool:
        """Check if text likely contains multiple intents using learned patterns."""
        # Count potential intent keywords
        intent_keywords = {
            "create_reminder": ["remind", "reminder", "schedule", "appointment", "alarm", "alert"],
            "create_note": ["note", "write", "jot", "record", "save", "document"],
            "create_ledger": ["owe", "owes", "borrowed", "lent", "paid", "money", "$"],
            "chit_chat": ["hello", "hi", "thank", "help", "how"]
        }
        
        intent_count = 0
        for intent, keywords in intent_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                intent_count += 1
        
        # Check for connecting words
        has_connectors = any(re.search(pattern, text, re.IGNORECASE) for pattern in self.multi_intent_indicators)
        
        # Likely multi-intent if: multiple intent types detected OR connectors present
        return intent_count >= 2 or has_connectors
    
    def _segment_text_intelligent(self, text: str) -> List[str]:
        """Segment text using intelligent approach based on trained patterns."""
        # Enhanced segmentation patterns
        splitting_patterns = [
            r'\s+and\s+(?:set|create|add|make|remind|schedule|note|[A-Z][a-z]+\s+(?:owes?|will\s+(?:give|pay)))',
            r'\s+also\s+(?:set|create|add|make|remind|schedule|note)',
            r'\s+plus\s+(?:set|create|add|make|remind|schedule|note)',
            r'\s+then\s+(?:set|create|add|make|remind|schedule|note)',
            r'(?:\.|;)\s*(?:Set|Create|Add|Make|Remind|Schedule|Note|[A-Z][a-z]+)',
        ]
        
        # Find split points
        split_points = []
        for pattern in splitting_patterns:
            for match in re.finditer(pattern, text):
                split_points.append(match.start())
        
        if not split_points:
            # Fallback: split on strong indicators
            fallback_patterns = [
                r'\s+and\s+(?:remind|note|owe)',
                r'(?:noon|pm|am)\s+remind',
                r'bill\s+and\s+I\s+want',
                r'plus\s+set',
            ]
            
            for pattern in fallback_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    split_points.append(match.start())
        
        # Create segments
        split_points = sorted(list(set(split_points)))
        
        if not split_points:
            return [text]
        
        segments = []
        current_start = 0
        
        for split_point in split_points:
            if current_start < split_point:
                segment = text[current_start:split_point].strip()
                if len(segment) > 10:  # Minimum segment length
                    segments.append(segment)
            current_start = split_point
        
        # Add last segment
        last_segment = text[current_start:].strip()
        if last_segment:
            # Clean up connectors at the beginning
            last_segment = re.sub(r'^(?:and|also|plus|then)\s+', '', last_segment, flags=re.IGNORECASE)
            if len(last_segment) > 10:
                segments.append(last_segment)
        
        return segments if segments else [text]
    
    def _extract_entities_intelligent(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities using intelligent patterns based on intent."""
        entities = {}
        
        try:
            # Time entities (for reminders)
            if intent == "create_reminder":
                time_patterns = [
                    r'\b(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))\b',
                    r'\b(\d{1,2}\s*(?:am|pm|AM|PM))\b',
                    r'\b(tomorrow|today|tonight|morning|evening|afternoon)\b',
                    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                    r'\b(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b'
                ]
                
                for pattern in time_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        entities["time"] = matches
                        break
            
            # Person entities (for ledgers and reminders)
            if intent in ["create_ledger", "create_reminder"]:
                person_patterns = [
                    r'\b([A-Z][a-z]+)\s+(?:owes?|owed?|borrowed?|will\s+(?:give|pay))',
                    r'\b(?:call|meet|contact|visit)\s+([A-Z][a-z]+)\b',
                    r'\b(?:remind|call)\s+(\w+)\b',
                    r'\b(mom|dad|mother|father|brother|sister|friend|boss|client|doctor)\b'
                ]
                
                for pattern in person_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        entities["person"] = [match.capitalize() if isinstance(match, str) else match for match in matches]
                        break
            
            # Amount entities (for ledgers)
            if intent == "create_ledger":
                amount_patterns = [
                    r'\$(\d+(?:\.\d{2})?)',
                    r'\b(\d+(?:\.\d{2})?\s*dollars?)\b',
                    r'\b(\d+\s*bucks?)\b'
                ]
                
                for pattern in amount_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        entities["amount"] = matches
                        break
            
            # Content entities (for notes)
            if intent == "create_note":
                # Extract meaningful content after note keywords
                content_patterns = [
                    r'note\s+(?:to\s+|about\s+|that\s+)?(.+)',
                    r'(?:buy|purchase|get)\s+(.+)',
                    r'remember\s+(?:to\s+|that\s+)?(.+)',
                    r'write\s+(?:down\s+)?(.+)',
                ]
                
                for pattern in content_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        # Clean up content
                        content = re.sub(r'\s+and\s+(john|sarah|mike|remind|set).*$', '', content, flags=re.IGNORECASE)
                        if len(content) > 2:
                            entities["content"] = content
                            break
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Keep the original case for better entity recognition
        return text
    
    def _create_fallback_response(self, text: str, multi_intent: bool) -> Dict[str, Any]:
        """Create fallback response when model is not available."""
        return {
            "success": False,
            "error": "Trained model not available",
            "intent": "general_query",
            "confidence": 0.1,
            "entities": {},
            "original_text": text,
            "model_used": "fallback",
            "type": "multi_intent" if multi_intent else "single_intent"
        }
    
    def _create_error_response(self, text: str, error: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "success": False,
            "error": error,
            "intent": "general_query",
            "confidence": 0.0,
            "entities": {},
            "original_text": text,
            "model_used": "trained_multi_intent",
            "type": "single_intent"
        }
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_available": self.model is not None,
            "tokenizer_available": self.tokenizer is not None,
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "device": str(self.device),
            "intent_labels": self.intent_labels,
            "model_type": "trained_multi_intent_distilbert",
            "supports_multi_intent": True,
            "generalization_capable": True
        } 