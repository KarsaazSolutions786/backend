"""
Intent Processor Service - Handles intent classification results and saves data to database
"""

import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from decimal import Decimal, InvalidOperation

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from connect_db import SessionLocal
from models.models import User, Reminder, Note, LedgerEntry, HistoryLog
from utils.logger import logger


class IntentProcessorService:
    """Service to process intent classification results and save to database."""
    
    def __init__(self):
        self.time_patterns = {
            'am_pm': re.compile(r'(\d{1,2}):?(\d{2})?\s*(a\.?m\.?|p\.?m\.?)', re.IGNORECASE),
            '24_hour': re.compile(r'(\d{1,2}):(\d{2})'),
            'relative': re.compile(r'(tomorrow|today|tonight|morning|evening|afternoon)', re.IGNORECASE),
            'in_x_time': re.compile(r'in\s+(\d+)\s*(minutes?|hours?|days?)', re.IGNORECASE)
        }
        
        self.amount_patterns = {
            'dollar': re.compile(r'\$(\d+(?:\.\d{2})?)', re.IGNORECASE),
            'number': re.compile(r'(\d+(?:\.\d{2})?)\s*(?:dollars?|bucks?)', re.IGNORECASE),
            'written': re.compile(r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:dollars?|bucks?)', re.IGNORECASE)
        }
        
        # Handler registry for intent processing
        self.intent_handlers = {
            "create_reminder": self._process_reminder,
            "create_note": self._process_note,
            "create_ledger": self._process_ledger,
            "add_expense": self._process_ledger,  # Route to ledger handler
            "chit_chat": self._process_chat,
            "general_query": self._process_chat,  # Route to chat handler
        }

    async def process_intent(self, intent_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Process intent classification result and save to appropriate database table.
        This method supports both single intent and multi-intent processing.
        
        Args:
            intent_data: Intent classification result from AI service
                        Can be single intent: {"intent": "create_reminder", "confidence": 0.95, ...}
                        Or multi-intent: {"intents": [{"type": "create_reminder", ...}, ...]}
            user_id: User ID from authentication
            
        Returns:
            Dictionary with operation result(s)
        """
        try:
            # Check if this is multi-intent or single intent
            if "intents" in intent_data and isinstance(intent_data["intents"], list):
                return await self.process_multi_intent(intent_data, user_id)
            else:
                return await self.process_single_intent(intent_data, user_id)
                
        except Exception as e:
            logger.error(f"Error processing intent data: {e}")
            return {
                'success': False,
                'error': f'Failed to process intent data: {str(e)}',
                'intent_data': intent_data
            }

    async def process_multi_intent(self, intent_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Process multiple intents from a single utterance with independent transactions.
        
        Args:
            intent_data: Multi-intent classification result with "intents" array
            user_id: User ID from authentication
            
        Returns:
            Dictionary with results array containing status of each intent processing
        """
        logger.info(f"Processing multi-intent for user: {user_id}")
        
        intents = intent_data.get("intents", [])
        original_text = intent_data.get("original_text", "")
        
        if not intents:
            return {
                'success': False,
                'error': 'No intents found in multi-intent data',
                'results': []
            }
        
        # Validate user exists once for all intents
        if not await self._validate_user(user_id):
            return {
                'success': False,
                'error': 'User not found',
                'results': []
            }
        
        results = []
        overall_success = True
        
        # Process each intent independently
        for i, intent_obj in enumerate(intents):
            intent_type = intent_obj.get("type", "unknown")
            confidence = intent_obj.get("confidence", 0.0)
            entities = intent_obj.get("entities", {})
            text_segment = intent_obj.get("text_segment", original_text)
            
            logger.info(f"Processing intent {i+1}/{len(intents)}: {intent_type} (confidence: {confidence:.2f})")
            
            try:
                # Create single intent data structure for processing
                single_intent_data = {
                    "intent": intent_type,
                    "confidence": confidence,
                    "entities": entities,
                    "original_text": text_segment
                }
                
                # Process using independent transaction
                result = await self._process_single_intent_with_transaction(
                    single_intent_data, user_id, text_segment
                )
                
                # Add intent info to result
                result["intent"] = intent_type
                result["text_segment"] = text_segment
                result["position"] = i + 1
                
                results.append(result)
                
                if not result.get("success", False):
                    overall_success = False
                    logger.warning(f"Intent {i+1} failed: {result.get('error', 'Unknown error')}")
                else:
                    logger.info(f"Intent {i+1} processed successfully")
                    
            except Exception as e:
                logger.error(f"Error processing intent {i+1} ({intent_type}): {e}")
                results.append({
                    "success": False,
                    "error": f"Failed to process intent: {str(e)}",
                    "intent": intent_type,
                    "text_segment": text_segment,
                    "position": i + 1
                })
                overall_success = False
        
        logger.info(f"Multi-intent processing completed: {len(results)} intents, overall_success: {overall_success}")
        
        return {
            "success": overall_success,
            "message": f"Processed {len(results)} intents" + (" (with some failures)" if not overall_success else " successfully"),
            "results": results,
            "total_intents": len(intents),
            "successful_intents": sum(1 for r in results if r.get("success", False)),
            "original_text": original_text
        }

    async def process_single_intent(self, intent_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Process single intent (backward compatibility method).
        
        Args:
            intent_data: Single intent classification result
            user_id: User ID from authentication
            
        Returns:
            Dictionary with operation result
        """
        intent = intent_data.get('intent', '').lower()
        original_text = intent_data.get('original_text', '')
        
        return await self._process_single_intent_with_transaction(intent_data, user_id, original_text)

    async def _process_single_intent_with_transaction(self, intent_data: Dict[str, Any], user_id: str, original_text: str) -> Dict[str, Any]:
        """
        Process a single intent with its own database transaction.
        
        Args:
            intent_data: Intent classification result from AI service
            user_id: User ID from authentication
            original_text: The text segment for this intent
            
        Returns:
            Dictionary with operation result
        """
        try:
            intent = intent_data.get('intent', '').lower()
            entities = intent_data.get('entities', {})
            confidence = intent_data.get('confidence', 0.0)
            
            logger.info(f"Processing single intent: {intent} for user: {user_id}")
            
            # Route to appropriate handler based on intent
            handler = self.intent_handlers.get(intent)
            if handler:
                return await handler(original_text, entities, user_id)
            else:
                # For unknown intents, save as chat interaction
                logger.warning(f"Unknown intent: {intent}, treating as chat")
                return await self._process_chat(original_text, entities, user_id)
                
        except Exception as e:
            logger.error(f"Error processing single intent: {e}")
            return {
                'success': False,
                'error': f'Failed to process intent: {str(e)}',
                'intent': intent_data.get('intent', 'unknown')
            }

    async def _validate_user(self, user_id: str) -> bool:
        """Validate that user exists in database."""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return user is not None
        except Exception as e:
            logger.error(f"Error validating user: {e}")
            return False
        finally:
            db.close()

    async def _process_reminder(self, original_text: str, entities: Dict, user_id: str) -> Dict[str, Any]:
        """Process create_reminder intent and save to reminders table."""
        db = SessionLocal()
        try:
            # Extract time information
            reminder_time = self._extract_time(original_text, entities)
            
            # Extract person information
            person = self._extract_person(entities, original_text)
            
            # Generate title and description
            title = self._generate_reminder_title(original_text, person)
            description = original_text.strip('"')
            
            # Create reminder object
            reminder = Reminder(
                id=uuid.uuid4(),
                user_id=user_id,
                title=title,
                description=description,
                time=reminder_time,
                created_by=user_id
            )
            
            db.add(reminder)
            db.commit()
            
            logger.info(f"Created reminder {reminder.id} for user {user_id}")
            
            return {
                'success': True,
                'message': 'Reminder created successfully',
                'data': {
                    'reminder_id': str(reminder.id),
                    'title': title,
                    'description': description,
                    'time': reminder_time.isoformat() if reminder_time else None,
                    'person': person
                },
                'intent': 'create_reminder'
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating reminder: {e}")
            return {
                'success': False,
                'error': f'Failed to create reminder: {str(e)}',
                'intent': 'create_reminder'
            }
        finally:
            db.close()

    async def _process_note(self, original_text: str, entities: Dict, user_id: str) -> Dict[str, Any]:
        """Process create_note intent and save to notes table."""
        db = SessionLocal()
        try:
            # Clean the original text
            content = original_text.strip('"')
            
            # Create note object
            note = Note(
                id=uuid.uuid4(),
                user_id=user_id,
                content=content,
                source='voice_input'
            )
            
            db.add(note)
            db.commit()
            
            logger.info(f"Created note {note.id} for user {user_id}")
            
            return {
                'success': True,
                'message': 'Note created successfully',
                'data': {
                    'note_id': str(note.id),
                    'content': content
                },
                'intent': 'create_note'
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating note: {e}")
            return {
                'success': False,
                'error': f'Failed to create note: {str(e)}',
                'intent': 'create_note'
            }
        finally:
            db.close()

    async def _process_ledger(self, original_text: str, entities: Dict, user_id: str) -> Dict[str, Any]:
        """Process create_ledger/add_expense intent and save to ledger_entries table."""
        db = SessionLocal()
        try:
            # Extract amount and person
            amount = self._extract_amount(original_text, entities)
            person = self._extract_person(entities, original_text)
            direction = self._extract_direction(original_text)
            
            if not amount:
                return {
                    'success': False,
                    'error': 'Could not extract amount from the text',
                    'intent': 'create_ledger'
                }
            
            # For standalone monetary amounts, make person optional
            # If no person is found, use a default placeholder
            if not person:
                person = "Unknown Contact"  # Default placeholder when no person found
                if self._is_standalone_amount(original_text):
                    logger.info(f"Standalone amount detected: '{original_text}', using Unknown Contact placeholder")
                else:
                    logger.info(f"No person found in '{original_text}', using Unknown Contact placeholder")
            
            # Create ledger entry
            ledger_entry = LedgerEntry(
                id=uuid.uuid4(),
                user_id=user_id,
                contact_name=person,
                amount=Decimal(str(amount)),
                direction=direction
            )
            
            db.add(ledger_entry)
            db.commit()
            
            logger.info(f"Created ledger entry {ledger_entry.id} for user {user_id}")
            
            # Generate appropriate description
            if person == "Unknown Contact":
                description = f"Amount: ${amount} (direction: {direction})"
            else:
                description = f"{person} {direction}{'s' if direction == 'owe' else 'd'} ${amount}"
            
            return {
                'success': True,
                'message': 'Ledger entry created successfully',
                'data': {
                    'ledger_id': str(ledger_entry.id),
                    'contact_name': person,
                    'amount': float(amount),
                    'direction': direction,
                    'description': description
                },
                'intent': 'create_ledger'
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating ledger entry: {e}")
            return {
                'success': False,
                'error': f'Failed to create ledger entry: {str(e)}',
                'intent': 'create_ledger'
            }
        finally:
            db.close()

    async def _process_chat(self, original_text: str, entities: Dict, user_id: str) -> Dict[str, Any]:
        """Process chit_chat intent and save to history_logs table."""
        db = SessionLocal()
        try:
            # Clean the original text
            content = original_text.strip('"')
            
            # Create history log entry
            history_log = HistoryLog(
                id=uuid.uuid4(),
                user_id=user_id,
                content=content,
                interaction_type='chit_chat'
            )
            
            db.add(history_log)
            db.commit()
            
            logger.info(f"Created chat log {history_log.id} for user {user_id}")
            
            return {
                'success': True,
                'message': 'Chat interaction logged successfully',
                'data': {
                    'log_id': str(history_log.id),
                    'content': content,
                    'interaction_type': 'chit_chat'
                },
                'intent': 'chit_chat'
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating chat log: {e}")
            return {
                'success': False,
                'error': f'Failed to log chat interaction: {str(e)}',
                'intent': 'chit_chat'
            }
        finally:
            db.close()

    def _extract_time(self, text: str, entities: Dict) -> Optional[datetime]:
        """Extract and parse time information from text and entities."""
        try:
            # First check entities for time information
            if 'time' in entities:
                time_entity = entities['time']
                if isinstance(time_entity, (list, tuple)) and len(time_entity) >= 2:
                    hour = int(time_entity[0]) if time_entity[0] else 0
                    minute = int(time_entity[1]) if time_entity[1] else 0
                    am_pm = time_entity[2] if len(time_entity) > 2 else None
                    
                    # Handle AM/PM
                    if am_pm and 'p' in am_pm.lower() and hour != 12:
                        hour += 12
                    elif am_pm and 'a' in am_pm.lower() and hour == 12:
                        hour = 0
                    
                    # Create datetime for today with the specified time
                    now = datetime.now()
                    reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If the time has passed today, schedule for tomorrow
                    if reminder_time <= now:
                        reminder_time += timedelta(days=1)
                    
                    return reminder_time
            
            # Try to parse time from text using regex patterns
            for pattern_name, pattern in self.time_patterns.items():
                match = pattern.search(text)
                if match:
                    if pattern_name == 'am_pm':
                        hour = int(match.group(1))
                        minute = int(match.group(2)) if match.group(2) else 0
                        am_pm = match.group(3).lower()
                        
                        if 'p' in am_pm and hour != 12:
                            hour += 12
                        elif 'a' in am_pm and hour == 12:
                            hour = 0
                        
                        now = datetime.now()
                        reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        
                        if reminder_time <= now:
                            reminder_time += timedelta(days=1)
                        
                        return reminder_time
                    
                    elif pattern_name == '24_hour':
                        hour = int(match.group(1))
                        minute = int(match.group(2))
                        
                        now = datetime.now()
                        reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        
                        if reminder_time <= now:
                            reminder_time += timedelta(days=1)
                        
                        return reminder_time
                    
                    elif pattern_name == 'relative':
                        relative_time = match.group(1).lower()
                        now = datetime.now()
                        
                        if relative_time == 'tomorrow':
                            return now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
                        elif relative_time == 'today':
                            return now.replace(hour=18, minute=0, second=0, microsecond=0)
                        elif relative_time == 'tonight':
                            return now.replace(hour=20, minute=0, second=0, microsecond=0)
                        elif relative_time == 'morning':
                            base_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
                            return base_time + timedelta(days=1) if base_time <= now else base_time
                        elif relative_time == 'afternoon':
                            base_time = now.replace(hour=14, minute=0, second=0, microsecond=0)
                            return base_time + timedelta(days=1) if base_time <= now else base_time
                        elif relative_time == 'evening':
                            base_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
                            return base_time + timedelta(days=1) if base_time <= now else base_time
                    
                    elif pattern_name == 'in_x_time':
                        number = int(match.group(1))
                        unit = match.group(2).lower()
                        
                        now = datetime.now()
                        if 'minute' in unit:
                            return now + timedelta(minutes=number)
                        elif 'hour' in unit:
                            return now + timedelta(hours=number)
                        elif 'day' in unit:
                            return now + timedelta(days=number)
            
            # Default to 1 hour from now if no time specified
            return datetime.now() + timedelta(hours=1)
            
        except Exception as e:
            logger.error(f"Error extracting time: {e}")
            return datetime.now() + timedelta(hours=1)

    def _extract_person(self, entities: Dict, text: str) -> Optional[str]:
        """Extract person/contact name from entities or text."""
        try:
            # Check entities first
            if 'person' in entities:
                return str(entities['person']).strip()
            
            # Enhanced regex patterns to find names in text
            name_patterns = [
                # Action + Name: "call John", "meet Sarah"
                r'\b(?:call|contact|meet|see|tell|remind)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                
                # Name + Owes: "John owes", "Sarah owed"
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owes?|owed?)\b',
                
                # Name + Will Give/Pay: "John will give", "Sarah will pay"
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will\s+)?(?:give|pay)\b',
                
                # Prepositions + Name: "to John", "with Sarah", "about Mike"
                r'\b(?:to|about|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                
                # Enhanced borrowing patterns: "Mike borrowed", "borrowed from John"
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+borrowed\b',
                r'\bborrowed\s+from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                
                # Lending patterns: "lent to John", "John lent"
                r'\blent\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+lent\b',
                
                # Record patterns: "record that John", "note that Sarah"
                r'\b(?:record|note)\s+that\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting person: {e}")
            return None

    def _extract_amount(self, text: str, entities: Dict) -> Optional[float]:
        """Extract monetary amount from text or entities."""
        try:
            # Check entities first
            if 'amount' in entities:
                try:
                    return float(entities['amount'])
                except (ValueError, TypeError):
                    pass
            
            # Use regex patterns to find amounts
            for pattern_name, pattern in self.amount_patterns.items():
                match = pattern.search(text)
                if match:
                    if pattern_name in ['dollar', 'number']:
                        return float(match.group(1))
                    elif pattern_name == 'written':
                        # Convert written numbers to digits
                        written_numbers = {
                            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                        }
                        written = match.group(1).lower()
                        if written in written_numbers:
                            return float(written_numbers[written])
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting amount: {e}")
            return None

    def _extract_direction(self, text: str) -> str:
        """Extract direction (owe/owed) from text."""
        text_lower = text.lower()
        
        # Patterns indicating someone owes the user money
        if any(phrase in text_lower for phrase in ['owes me', 'borrowed from me', 'lent to']):
            return 'owed'  # They owe me
        
        # Patterns indicating user owes someone money
        if any(phrase in text_lower for phrase in ['i owe', 'borrowed from', 'lent me']):
            return 'owe'  # I owe them
        
        # Default assumption based on common phrasing
        if 'owe' in text_lower:
            return 'owed'  # Most common case: "John owes me $50"
        
        return 'owe'  # Default

    def _generate_reminder_title(self, text: str, person: Optional[str]) -> str:
        """Generate a concise title for the reminder."""
        try:
            # Clean the text
            clean_text = text.strip('"').lower()
            
            # Extract key action words
            action_words = ['call', 'meet', 'contact', 'email', 'text', 'visit', 'pick up', 'buy', 'do']
            
            for action in action_words:
                if action in clean_text:
                    if person:
                        return f"{action.capitalize()} {person}"
                    else:
                        # Extract the object after the action
                        parts = clean_text.split(action, 1)
                        if len(parts) > 1:
                            obj = parts[1].strip().split()[0] if parts[1].strip() else ''
                            return f"{action.capitalize()} {obj}".strip()
                        return action.capitalize()
            
            # If no action word found, create a generic title
            if person:
                return f"Reminder about {person}"
            
            # Use first few words as title
            words = clean_text.split()[:4]
            return ' '.join(words).capitalize()
            
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "Reminder"

    def _is_standalone_amount(self, text: str) -> bool:
        """Check if the text appears to be a standalone amount."""
        import re
        
        text_clean = text.strip()
        
        # Patterns for standalone monetary amounts (no context words)
        standalone_patterns = [
            r'^\$\d+(?:,\d{3})*(?:\.\d{2})?$',  # $1000, $1,000, $1000.00
            r'^\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|bucks?)$',  # 1000 dollars
            r'^€\d+(?:,\d{3})*(?:\.\d{2})?$',   # €1000
            r'^£\d+(?:,\d{3})*(?:\.\d{2})?$',   # £1000
        ]
        
        # Check if text matches standalone amount pattern
        for pattern in standalone_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Additional check: if text contains only amount-related words
        words = text_clean.lower().split()
        amount_words = {'dollars', 'dollar', 'bucks', 'buck', 'usd', '$'}
        
        # If all words are amount-related or numbers, it's standalone
        if len(words) <= 3 and all(
            word.replace('$', '').replace(',', '').replace('.', '').isdigit() or 
            word in amount_words or
            word.startswith('$')
            for word in words
        ):
            return True
            
        return False 