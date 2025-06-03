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

# Placeholder for a proper i18n/l10n library
# For example, using `python-i18n` or `fluent`
# from i18n import t
def get_localized_string(key: str, lang: str, **kwargs) -> str:
    """Placeholder for actual translation.
    In a real app, this would use an i18n library.
    """
    # Simple English fallbacks for now
    translations_en = {
        "reminder_created_successfully": "Reminder created successfully",
        "note_created_successfully": "Note created successfully",
        "ledger_entry_created_successfully": "Ledger entry created successfully",
        "chat_interaction_logged_successfully": "Chat interaction logged successfully",
        "failed_to_create_reminder": "Failed to create reminder: {error}",
        "failed_to_create_note": "Failed to create note: {error}",
        "failed_to_create_ledger_entry": "Failed to create ledger entry: {error}",
        "failed_to_log_chat_interaction": "Failed to log chat interaction: {error}",
        "could_not_extract_amount": "Could not extract amount from the text",
        "user_not_found": "User not found",
        "no_intents_found": "No intents found in multi-intent data",
        "processed_intents_message": "Processed {count} intents successfully",
        "processed_intents_with_failures_message": "Processed {count} intents (with some failures)",
        "unknown_contact": "Unknown Contact",
        "ledger_description_amount_direction": "Amount: ${amount} (direction: {direction})",
        "ledger_description_person_direction": "{person} {direction_verb} ${amount}",
        "reminder_title_about_person": "Reminder about {person}",
        "default_reminder_title": "Reminder"
    }
    # This would be expanded for all supported languages
    # translations_es = { ... } 
    
    if lang == "en": # Add more languages here
        return translations_en.get(key, key).format(**kwargs)
    return key.format(**kwargs) # Fallback to key if lang not supported


class IntentProcessorService:
    """Service to process intent classification results and save to database."""
    
    def __init__(self):
        # These patterns are English-specific and should be moved to a lang-specific resource loader
        # or made language-aware if kept here. For now, they are illustrative.
        self.time_patterns_en = { 
            'am_pm': re.compile(r'(\d{1,2}):?(\d{2})?\s*(a\.?m\.?|p\.?m\.?)', re.IGNORECASE),
            '24_hour': re.compile(r'(\d{1,2}):(\d{2})'),
            'relative': re.compile(r'(tomorrow|today|tonight|morning|evening|afternoon)', re.IGNORECASE),
            'in_x_time': re.compile(r'in\s+(\d+)\s*(minutes?|hours?|days?)', re.IGNORECASE)
        }
        
        self.amount_patterns_en = {
            'dollar': re.compile(r'\$(\d+(?:\.\d{2})?)', re.IGNORECASE),
            'number': re.compile(r'(\d+(?:\.\d{2})?)\s*(?:dollars?|bucks?)', re.IGNORECASE),
            'written': re.compile(r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:dollars?|bucks?)', re.IGNORECASE)
        }
        
        self.action_words_en = ['call', 'meet', 'contact', 'email', 'text', 'visit', 'pick up', 'buy', 'do']

        self.ledger_direction_keywords_en = {
            'owed': ['owes me', 'borrowed from me', 'lent to', 'will give me', 'will pay me'], # They owe user
            'owe': ['i owe', 'borrowed from', 'lent me', 'i will give', 'i will pay'] # User owes them
        }


        # Handler registry for intent processing
        self.intent_handlers = {
            "create_reminder": self._process_reminder,
            "create_note": self._process_note,
            "create_ledger": self._process_ledger,
            "add_expense": self._process_ledger,
            "chit_chat": self._process_chat,
            "general_query": self._process_chat,
        }

    async def process_intent(self, intent_data: Dict[str, Any], user_id: str, language_code: str = "en") -> Dict[str, Any]:
        logger.info(f"Processing intent with language: {language_code}")
        try:
            if "intents" in intent_data and isinstance(intent_data["intents"], list):
                return await self.process_multi_intent(intent_data, user_id, language_code)
            else:
                return await self.process_single_intent(intent_data, user_id, language_code)
        except Exception as e:
            logger.error(f"Error processing intent data ({language_code}): {e}")
            return {'success': False, 'error': f'Failed to process intent data: {str(e)}', 'intent_data': intent_data}

    async def process_multi_intent(self, intent_data: Dict[str, Any], user_id: str, language_code: str = "en") -> Dict[str, Any]:
        logger.info(f"Processing multi-intent for user: {user_id} (lang: {language_code})")
        intents = intent_data.get("intents", [])
        original_text = intent_data.get("original_text", "")
        
        if not intents:
            return {'success': False, 'error': get_localized_string("no_intents_found", language_code), 'results': []}
        
        if not await self._validate_user(user_id):
            return {'success': False, 'error': get_localized_string("user_not_found", language_code), 'results': []}
        
        results = []
        overall_success = True
        
        for i, intent_obj in enumerate(intents):
            intent_type = intent_obj.get("type", "unknown")
            entities = intent_obj.get("entities", {})
            text_segment = intent_obj.get("text_segment", original_text)
            
            logger.info(f"Processing intent {i+1}/{len(intents)}: {intent_type} (lang: {language_code})")
            try:
                single_intent_data = {"intent": intent_type, "entities": entities, "original_text": text_segment}
                result = await self._process_single_intent_with_transaction(single_intent_data, user_id, text_segment, language_code)
                result["intent"] = intent_type; result["text_segment"] = text_segment; result["position"] = i + 1
                results.append(result)
                if not result.get("success", False): overall_success = False
            except Exception as e:
                logger.error(f"Error processing intent {i+1} ({intent_type}, lang: {language_code}): {e}")
                results.append({"success": False, "error": str(e), "intent": intent_type, "text_segment": text_segment, "position": i + 1})
                overall_success = False
        
        count = len(results)
        success_count = sum(1 for r in results if r.get("success", False))
        message_key = "processed_intents_message" if overall_success else "processed_intents_with_failures_message"
        message = get_localized_string(message_key, language_code, count=count)

        return {"success": overall_success, "message": message, "results": results, "total_intents": count, "successful_intents": success_count, "original_text": original_text}

    async def process_single_intent(self, intent_data: Dict[str, Any], user_id: str, language_code: str = "en") -> Dict[str, Any]:
        original_text = intent_data.get('original_text', '')
        return await self._process_single_intent_with_transaction(intent_data, user_id, original_text, language_code)

    async def _process_single_intent_with_transaction(self, intent_data: Dict[str, Any], user_id: str, original_text: str, language_code: str = "en") -> Dict[str, Any]:
        try:
            intent = intent_data.get('intent', '').lower()
            entities = intent_data.get('entities', {})
            logger.info(f"Processing single intent: {intent} for user: {user_id} (lang: {language_code})")
            
            handler = self.intent_handlers.get(intent)
            if handler:
                # Pass language_code to the specific handler
                return await handler(original_text, entities, user_id, language_code)
            else:
                logger.warning(f"Unknown intent: {intent}, treating as chat (lang: {language_code})")
                return await self._process_chat(original_text, entities, user_id, language_code)
        except Exception as e:
            logger.error(f"Error processing single intent ({language_code}): {e}")
            return {'success': False, 'error': str(e), 'intent': intent_data.get('intent', 'unknown')}

    async def _validate_user(self, user_id: str) -> bool: # No lang change needed
        db = SessionLocal()
        try: return db.query(User).filter(User.id == user_id).first() is not None
        finally: db.close()

    async def _process_reminder(self, original_text: str, entities: Dict, user_id: str, language_code: str = "en") -> Dict[str, Any]:
        db = SessionLocal()
        try:
            reminder_time = self._extract_time(original_text, entities, language_code)
            person = self._extract_person(entities, original_text, language_code)
            title = self._generate_reminder_title(original_text, person, language_code)
            description = original_text.strip('"')
            
            reminder = Reminder(id=uuid.uuid4(), user_id=user_id, title=title, description=description, time=reminder_time, created_by=user_id)
            db.add(reminder); db.commit()
            logger.info(f"Created reminder {reminder.id} for user {user_id} (lang: {language_code})")
            
            return {
                'success': True, 
                'message': get_localized_string("reminder_created_successfully", language_code),
                'data': {'reminder_id': str(reminder.id), 'title': title, 'description': description, 'time': reminder_time.isoformat() if reminder_time else None, 'person': person},
                'intent': 'create_reminder'
            }
        except Exception as e:
            db.rollback(); logger.error(f"Error creating reminder ({language_code}): {e}")
            return {'success': False, 'error': get_localized_string("failed_to_create_reminder", language_code, error=str(e)), 'intent': 'create_reminder'}
        finally: db.close()

    async def _process_note(self, original_text: str, entities: Dict, user_id: str, language_code: str = "en") -> Dict[str, Any]:
        db = SessionLocal()
        try:
            content = original_text.strip('"')
            note = Note(id=uuid.uuid4(), user_id=user_id, content=content, source='voice_input')
            db.add(note); db.commit()
            logger.info(f"Created note {note.id} for user {user_id} (lang: {language_code})")
            return {
                'success': True, 'message': get_localized_string("note_created_successfully", language_code),
                'data': {'note_id': str(note.id), 'content': content}, 'intent': 'create_note'
            }
        except Exception as e:
            db.rollback(); logger.error(f"Error creating note ({language_code}): {e}")
            return {'success': False, 'error': get_localized_string("failed_to_create_note", language_code, error=str(e)), 'intent': 'create_note'}
        finally: db.close()

    async def _process_ledger(self, original_text: str, entities: Dict, user_id: str, language_code: str = "en") -> Dict[str, Any]:
        db = SessionLocal()
        try:
            amount = self._extract_amount(original_text, entities, language_code)
            person = self._extract_person(entities, original_text, language_code)
            direction = self._extract_direction(original_text, language_code)
            
            if not amount:
                return {'success': False, 'error': get_localized_string("could_not_extract_amount", language_code), 'intent': 'create_ledger'}
            
            if not person:
                person_default_key = "unknown_contact"
                person = get_localized_string(person_default_key, language_code)
                if self._is_standalone_amount(original_text, language_code):
                    logger.info(f"Standalone amount detected ('{language_code}'): '{original_text}', using '{person}'")
                else:
                    logger.info(f"No person found in '{original_text}' ('{language_code}'), using '{person}'")
            
            ledger_entry = LedgerEntry(id=uuid.uuid4(),user_id=user_id,contact_name=person,amount=Decimal(str(amount)),direction=direction)
            db.add(ledger_entry); db.commit()
            logger.info(f"Created ledger entry {ledger_entry.id} for user {user_id} (lang: {language_code})")
            
            desc_key = "ledger_description_amount_direction" if person == get_localized_string("unknown_contact", language_code) else "ledger_description_person_direction"
            # For "{person} {direction_verb} ${amount}", direction_verb needs localization (e.g. owes/owed vs. debe/debía)
            # This is a simplified example.
            direction_verb = direction # Placeholder, needs proper localization
            if language_code == 'en': # Simple case for english
                direction_verb = f"{direction}{'s' if direction == 'owe' else 'd'}"

            description = get_localized_string(desc_key, language_code, person=person, amount=amount, direction=direction, direction_verb=direction_verb)
            
            return {
                'success': True, 'message': get_localized_string("ledger_entry_created_successfully", language_code),
                'data': {'ledger_id': str(ledger_entry.id), 'contact_name': person, 'amount': float(amount), 'direction': direction, 'description': description},
                'intent': 'create_ledger'
            }
        except Exception as e:
            db.rollback(); logger.error(f"Error creating ledger entry ({language_code}): {e}")
            return {'success': False, 'error': get_localized_string("failed_to_create_ledger_entry", language_code, error=str(e)), 'intent': 'create_ledger'}
        finally: db.close()

    async def _process_chat(self, original_text: str, entities: Dict, user_id: str, language_code: str = "en") -> Dict[str, Any]:
        db = SessionLocal()
        try:
            content = original_text.strip('"')
            history_log = HistoryLog(id=uuid.uuid4(),user_id=user_id,content=content,interaction_type='chit_chat')
            db.add(history_log); db.commit()
            logger.info(f"Created chat log {history_log.id} for user {user_id} (lang: {language_code})")
            return {
                'success': True, 'message': get_localized_string("chat_interaction_logged_successfully", language_code),
                'data': {'log_id': str(history_log.id), 'content': content, 'interaction_type': 'chit_chat'},
                'intent': 'chit_chat'
            }
        except Exception as e:
            db.rollback(); logger.error(f"Error creating chat log ({language_code}): {e}")
            return {'success': False, 'error': get_localized_string("failed_to_log_chat_interaction", language_code, error=str(e)), 'intent': 'chit_chat'}
        finally: db.close()

    def _extract_time(self, text: str, entities: Dict, language_code: str = "en") -> Optional[datetime]:
        # This method needs significant localization for time expressions, date words, AM/PM markers, etc.
        # Using English patterns as a placeholder.
        # For a real app, a library like `dateparser` with language support would be better.
        logger.debug(f"Extracting time from '{text}' with entities {entities} (lang: {language_code})")
        time_patterns_to_use = self.time_patterns_en # Should be self.time_patterns.get(language_code, self.time_patterns_en)

        try:
            if 'time' in entities: # Entities are already extracted by IntentService (language-aware)
                time_entity = entities['time'] # This could be a string or structured data
                # Parsing logic here needs to be robust and potentially language-aware if time_entity is a string
                # For demo, assume time_entity is a tuple like ('10', '00', 'pm') from IntentService
                if isinstance(time_entity, (list, tuple)) and len(time_entity) >= 2:
                    hour = int(time_entity[0]) if time_entity[0] else 0
                    minute = int(time_entity[1]) if time_entity[1] else 0
                    am_pm = time_entity[2].lower() if len(time_entity) > 2 and time_entity[2] else None
                    
                    if am_pm and ('p' in am_pm or 'pm' in am_pm) and hour != 12: hour += 12
                    elif am_pm and ('a' in am_pm or 'am' in am_pm) and hour == 12: hour = 0
                    
                    now = datetime.now()
                    reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    return reminder_time + timedelta(days=1) if reminder_time <= now else reminder_time
            
            # Fallback to regex parsing if entities don't provide structured time
            # This part is highly English-centric
            if language_code == "en":
                for pattern_name, pattern in time_patterns_to_use.items():
                    match = pattern.search(text)
                    if match:
                        # ... (existing English time parsing logic) ...
                        # This would need to be heavily localized or replaced
                        if pattern_name == 'am_pm':
                            hour = int(match.group(1)); minute = int(match.group(2)) if match.group(2) else 0
                            am_pm = match.group(3).lower()
                            if 'p' in am_pm and hour != 12: hour += 12
                            elif 'a' in am_pm and hour == 12: hour = 0
                            now = datetime.now()
                            reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            return reminder_time + timedelta(days=1) if reminder_time <= now else reminder_time
                        # ... other patterns ...
            
            logger.warning(f"Could not extract specific time for '{text}' (lang: {language_code}), defaulting.")
            return datetime.now() + timedelta(hours=1) # Default
        except Exception as e:
            logger.error(f"Error extracting time ({language_code}): {e}")
            return datetime.now() + timedelta(hours=1)

    def _extract_person(self, entities: Dict, text: str, language_code: str = "en") -> Optional[str]:
        logger.debug(f"Extracting person from '{text}' with entities {entities} (lang: {language_code})")
        if 'person' in entities and entities['person']: # Entity from IntentService
            return str(entities['person']).strip()
        
        # Fallback if IntentService didn't find a person - this is English-centric
        # In a real multilingual app, rely on IntentService's language-aware NER
        if language_code == "en":
            name_patterns_en = [
                r'\b(?:call|contact|meet|see|tell|remind)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:owes?|owed?)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will\s+)?(?:give|pay)\b',
                r'\b(?:to|about|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            ]
            for pattern in name_patterns_en:
                match = re.search(pattern, text)
                if match: return match.group(1).strip()
        return None

    def _extract_amount(self, text: str, entities: Dict, language_code: str = "en") -> Optional[float]:
        logger.debug(f"Extracting amount from '{text}' with entities {entities} (lang: {language_code})")
        if 'amount' in entities and entities['amount']: # Entity from IntentService
            try:
                # Amount from IntentService might be a string like "50" or "1,000.50"
                # Need to handle different decimal/thousand separators based on language_code
                amount_str = str(entities['amount'])
                if language_code in ['es', 'fr', 'de', 'it', 'pt', 'pl', 'ru', 'nl', 'cs']: # Example, uses comma as decimal
                    amount_str = amount_str.replace('.', '').replace(',', '.') 
                else: # Default (e.g. English)
                    amount_str = amount_str.replace(',', '')
                return float(amount_str)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert entity amount '{entities['amount']}' to float ({language_code}): {e}")
        
        # Fallback - English-centric regex
        if language_code == "en":
            amount_patterns_to_use = self.amount_patterns_en
            for pattern_name, pattern in amount_patterns_to_use.items():
                match = pattern.search(text)
                if match:
                    if pattern_name in ['dollar', 'number']: return float(match.group(1))
                    # ... existing written number logic (highly English specific) ...
        
        logger.warning(f"Could not extract specific amount for '{text}' (lang: {language_code}).")
        return None

    def _extract_direction(self, text: str, language_code: str = "en") -> str:
        # This is highly language-specific.
        logger.debug(f"Extracting direction from '{text}' (lang: {language_code})")
        text_lower = text.lower()
        
        if language_code == "en": # English specific keywords
            if any(phrase in text_lower for phrase in self.ledger_direction_keywords_en.get('owed',[])): return 'owed'
            if any(phrase in text_lower for phrase in self.ledger_direction_keywords_en.get('owe',[])): return 'owe'
            if 'owe' in text_lower: return 'owed' # Default for English "owe"
        
        # Add other language logic here...
        # E.g., for Spanish: "me debe" -> owed, "debo a" -> owe
        
        logger.warning(f"Could not determine ledger direction for '{text}' (lang: {language_code}), defaulting to 'owe'")
        return 'owe' 

    def _generate_reminder_title(self, text: str, person: Optional[str], language_code: str = "en") -> str:
        # Title generation is also language-specific.
        logger.debug(f"Generating title for '{text}', person: {person} (lang: {language_code})")
        
        if language_code == "en": # English specific logic
            clean_text = text.strip('"').lower()
            action_words = self.action_words_en
            for action in action_words:
                if action in clean_text:
                    if person: return f"{action.capitalize()} {person}"
                    parts = clean_text.split(action, 1)
                    if len(parts) > 1:
                        obj = parts[1].strip().split()[0] if parts[1].strip() else ''
                        return f"{action.capitalize()} {obj}".strip()
                    return action.capitalize()
            if person: return get_localized_string("reminder_title_about_person", language_code, person=person)
            words = clean_text.split()[:4]
            return ' '.join(words).capitalize() if words else get_localized_string("default_reminder_title", language_code)

        # Fallback for other languages - just use first few words or a generic title
        default_title = get_localized_string("default_reminder_title", language_code)
        if person : return get_localized_string("reminder_title_about_person", language_code, person=person)
        
        words = text.strip('"').split()[:4]
        return ' '.join(words).capitalize() if words else default_title


    def _is_standalone_amount(self, text: str, language_code: str = "en") -> bool:
        # This also needs to be language-aware for currency symbols and number formats.
        # The patterns are now in IntentService, but this method could do extra checks.
        logger.debug(f"Checking if '{text}' is standalone amount (lang: {language_code})")
        # This check should ideally use localized patterns for amounts,
        # or rely on the entity extraction from IntentService being robust enough.
        # For now, this is a simplified placeholder.
        
        # A very basic check, assuming IntentService did the heavy lifting for entity patterns
        # If entities are extracted and the text segment is ONLY the amount, it might be standalone.
        # This is hard to determine reliably without full context or more advanced parsing.
        
        if language_code == "en": # English-specific quick checks
            text_clean = text.strip()
            standalone_patterns_en = [
                r'^\$\d+(?:,\d{3})*(?:\.\d{2})?$',
                r'^\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|bucks?)$',
            ]
            for pattern in standalone_patterns_en:
                if re.match(pattern, text_clean, re.IGNORECASE): return True
        
        # Add more language specific checks or rely on a more robust system
        return False 