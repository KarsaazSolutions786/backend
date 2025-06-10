"""
Database Integration Service for Intent Classification Results
Handles storing classification results into appropriate PostgreSQL tables.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import re
import os

# Import logger first
from utils.logger import logger

try:
    from sqlalchemy.orm import Session
    from sqlalchemy import text, select, insert, update
    from sqlalchemy.exc import SQLAlchemyError
    
    # Import the existing database connection
    from connect_db import SessionLocal, get_db
    
    DATABASE_AVAILABLE = True
    logger.info("Database connection available")
except ImportError as e:
    # Fallback when database dependencies are not available
    DATABASE_AVAILABLE = False
    
    # Mock classes for testing
    class Session:
        pass
    
    class SQLAlchemyError(Exception):
        pass
    
    def text(query):
        return query
    
    def get_db():
        """Mock database session for testing."""
        return None

    logger.warning(f"Database dependencies not available: {e}")

class DatabaseIntegrationService:
    """
    Service to integrate intent classification results with database storage.
    Handles both single and multi-intent classifications.
    """
    
    def __init__(self):
        # Updated table schemas to match the actual database structure
        self.table_schemas = {
            "reminders": {
                "required_fields": ["title", "user_id"],
                "optional_fields": ["time", "description", "repeat_pattern", "timezone", "is_shared", "created_by", "created_at"],
                "default_values": {
                    "repeat_pattern": "none", 
                    "timezone": "UTC", 
                    "is_shared": False, 
                    "created_by": None,
                    "created_at": datetime.now
                }
            },
            "notes": {
                "required_fields": ["content", "user_id"],
                "optional_fields": ["source", "created_at"],
                "default_values": {"source": "ai_pipeline", "created_at": datetime.now}
            },
            "ledger_entries": {
                "required_fields": ["user_id"],
                "optional_fields": ["amount", "contact_name", "direction", "created_at"],
                "default_values": {"amount": 0.0, "direction": "owe", "created_at": datetime.now}
            },
            "history_logs": {
                "required_fields": ["content", "user_id"],
                "optional_fields": ["interaction_type", "created_at"],
                "default_values": {"interaction_type": "ai_chat", "created_at": datetime.now}
            }
        }
        
        # Mock services for now - in production these would be real service imports
        self.reminder_service = None
        self.note_service = None
        self.ledger_service = None
        self.history_service = None
        
        logger.info("Database Integration Service initialized")
    
    async def store_classification_result(self, 
                                        classification_result: Dict[str, Any], 
                                        user_id: str) -> Dict[str, Any]:
        """
        Store intent classification result(s) in the database.
        
        Args:
            classification_result: Result from intent classification
            user_id: User ID for database records
            
        Returns:
            Dictionary with storage results
        """
        try:
            if classification_result.get("type") == "multi_intent":
                return await self._store_multi_intent_results(classification_result, user_id)
            else:
                return await self._store_single_intent_result(classification_result, user_id)
        except Exception as e:
            logger.error(f"Failed to store classification result: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    async def _store_single_intent_result(self, 
                                        result: Dict[str, Any], 
                                        user_id: str) -> Dict[str, Any]:
        """Store a single intent classification result."""
        try:
            intent = result["intent"]
            entities = result.get("entities", {})
            
            # Prepare data for database
            db_record = await self._prepare_db_record(intent, entities, result, user_id)
            
            if not db_record:
                return {
                    "success": False,
                    "error": f"Could not prepare database record for intent: {intent}",
                    "intent": intent
                }
            
            # Store in database
            stored_id = await self._insert_record(db_record["table"], db_record["data"])
            
            if stored_id:
                logger.info(f"Successfully stored {intent} record with ID: {stored_id}")
                return {
                    "success": True,
                    "intent": intent,
                    "table": db_record["table"],
                    "record_id": stored_id,
                    "data": db_record["data"]
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to insert record into database",
                    "intent": intent
                }
                
        except Exception as e:
            logger.error(f"Error storing single intent result: {e}")
            return {
                "success": False,
                "error": str(e),
                "intent": result.get("intent", "unknown")
            }
    
    async def _store_multi_intent_results(self, 
                                        result: Dict[str, Any], 
                                        user_id: str) -> Dict[str, Any]:
        """Store multiple intent classification results."""
        try:
            results = result.get("results", [])
            storage_results = []
            successful_stores = 0
            
            for i, intent_result in enumerate(results):
                logger.info(f"Storing intent {i+1}/{len(results)}: {intent_result.get('intent')}")
                
                store_result = await self._store_single_intent_result(intent_result, user_id)
                storage_results.append(store_result)
                
                if store_result["success"]:
                    successful_stores += 1
            
            return {
                "success": successful_stores > 0,
                "type": "multi_intent",
                "total_intents": len(results),
                "successful_stores": successful_stores,
                "failed_stores": len(results) - successful_stores,
                "results": storage_results,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error storing multi-intent results: {e}")
            return {
                "success": False,
                "error": str(e),
                "type": "multi_intent"
            }
    
    async def _prepare_db_record(self, 
                               intent: str, 
                               entities: Dict[str, Any], 
                               result: Dict[str, Any], 
                               user_id: str) -> Optional[Dict[str, Any]]:
        """Prepare database record based on intent type."""
        try:
            if intent == "create_reminder":
                return await self._prepare_reminder_record(entities, result, user_id)
            elif intent == "create_note":
                return await self._prepare_note_record(entities, result, user_id)
            elif intent == "create_ledger":
                return await self._prepare_ledger_record(entities, result, user_id)
            elif intent == "chit_chat":
                return await self._prepare_history_record(entities, result, user_id)
            else:
                logger.warning(f"Unknown intent type: {intent}")
                return await self._prepare_history_record(entities, result, user_id)
                
        except Exception as e:
            logger.error(f"Error preparing database record: {e}")
            return None
    
    def _generate_reminder_title(self, text: str, entities: Dict[str, Any]) -> str:
        """Generate a concise title for the reminder based on text content and entities."""
        try:
            # Clean the text
            clean_text = text.strip('"').lower()
            
            # Remove common reminder prefixes to get to the core action
            prefixes_to_remove = [
                r'^set\s+(?:a\s+)?reminder\s+to\s+',
                r'^remind\s+me\s+to\s+',
                r'^create\s+(?:a\s+)?reminder\s+to\s+',
                r'^make\s+(?:a\s+)?reminder\s+to\s+',
                r'^add\s+(?:a\s+)?reminder\s+to\s+'
            ]
            
            for prefix in prefixes_to_remove:
                clean_text = re.sub(prefix, '', clean_text, flags=re.IGNORECASE).strip()
            
            # Remove time information to focus on action only
            time_patterns = [
                r'\s+at\s+\d+(?::\d+)?\s*(?:am|pm|a\.m\.|p\.m\.)?.*$',  # " at 7 00 p.m"
                r'\s+on\s+\w+.*$',                                      # " on Monday"
                r'\s+tomorrow.*$',                                      # " tomorrow"
                r'\s+today.*$',                                         # " today"
                r'\s+tonight.*$',                                       # " tonight"
                r'\s+in\s+\d+.*$'                                       # " in 5 minutes"
            ]
            
            for pattern in time_patterns:
                clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE).strip()
            
            # Extract person from entities if available
            person = None
            if entities.get("person"):
                if isinstance(entities["person"], list) and entities["person"]:
                    person = entities["person"][0]
                elif isinstance(entities["person"], str):
                    person = entities["person"]
            
            # Extract key action words and create title
            action_words = ['call', 'meet', 'contact', 'email', 'text', 'visit', 'pick up', 'buy', 'do', 'go', 'book', 'schedule']
            
            for action in action_words:
                if action in clean_text:
                    if person:
                        return f"{action.capitalize()} {person}"
                    else:
                        # Extract the object after the action
                        parts = clean_text.split(action, 1)
                        if len(parts) > 1:
                            # Get the meaningful part after the action
                            remaining = parts[1].strip()
                            # Remove common words and extract core object
                            remaining = re.sub(r'^(?:to\s+|the\s+|a\s+|an\s+)', '', remaining)
                            obj_words = remaining.split()[:2]  # Take first 2 words for brevity
                            if obj_words:
                                obj = ' '.join(obj_words)
                                return f"{action.capitalize()} {obj}".strip()
                        return action.capitalize()
            
            # Handle specific common patterns
            if 'go' in clean_text:
                # Extract destination (remove time/location extras)
                go_match = re.search(r'go\s+(?:to\s+)?(\w+)', clean_text)
                if go_match:
                    destination = go_match.group(1).capitalize()
                    return f"Go to {destination}"
                return "Go somewhere"
            
            if 'book' in clean_text:
                # Extract what to book
                book_match = re.search(r'book\s+(?:a\s+)?(\w+)', clean_text)
                if book_match:
                    item = book_match.group(1).capitalize()
                    return f"Book {item}"
                return "Book something"
            
            # If person found but no specific action, create generic reminder
            if person:
                return f"Reminder about {person}"
            
            # Use first few meaningful words as title, removing common filler words
            words = clean_text.split()
            meaningful_words = []
            filler_words = {'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'and', 'or'}
            
            for word in words[:4]:  # Check first 4 words
                if word not in filler_words and len(word) > 2:
                    meaningful_words.append(word)
                if len(meaningful_words) >= 2:  # Stop after 2 meaningful words for brevity
                    break
            
            if meaningful_words:
                return ' '.join(meaningful_words).capitalize()
            
            # Final fallback
            return "Reminder"
            
        except Exception as e:
            logger.error(f"Error generating reminder title: {e}")
            return "Reminder"
    
    async def _prepare_reminder_record(self, 
                                     entities: Dict[str, Any], 
                                     result: Dict[str, Any], 
                                     user_id: str) -> Dict[str, Any]:
        """Prepare reminder record for database."""
        
        # Get the segment text if available (for multi-intent), otherwise use original text
        segment_text = result.get("segment", result.get("original_text", ""))
        
        # Generate a clean title using the segment text
        title = self._generate_reminder_title(segment_text, entities)
        
        data = {
            "id": str(uuid.uuid4()),  # Generate UUID
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now()
        }
        
        # Extract and parse time if available
        time_entities = entities.get("time", [])
        if time_entities:
            # Try to parse the first time entity into a proper timestamp
            time_str = time_entities[0] if isinstance(time_entities, list) else str(time_entities)
            parsed_time = self._parse_time_string(time_str)
            if parsed_time:
                data["time"] = parsed_time
        
        # Store the segment text as description (clean text without full transcription)
        data["description"] = segment_text
        
        return {
            "table": "reminders",
            "data": data
        }
    
    def _parse_time_string(self, time_str: str) -> Optional[datetime]:
        """Parse time string into a datetime object."""
        try:
            time_str = time_str.lower().strip()
            now = datetime.now()
            
            # Handle common time formats
            if 'pm' in time_str or 'am' in time_str:
                # Extract hour from "5 pm", "at 5 pm", etc.
                hour_match = re.search(r'(\d{1,2})', time_str)
                if hour_match:
                    hour = int(hour_match.group(1))
                    if 'pm' in time_str and hour != 12:
                        hour += 12
                    elif 'am' in time_str and hour == 12:
                        hour = 0
                    
                    # Create datetime for today at the specified hour
                    reminder_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # If the time has already passed today, set it for tomorrow
                    if reminder_time <= now:
                        reminder_time += timedelta(days=1)
                    
                    return reminder_time
            
            # Handle "tomorrow" 
            elif 'tomorrow' in time_str:
                return now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
            # Handle "next week"
            elif 'next week' in time_str:
                return now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=7)
            
            # If we can't parse it, return None (field is nullable)
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse time string '{time_str}': {e}")
            return None
    
    async def _prepare_note_record(self, 
                                 entities: Dict[str, Any], 
                                 result: Dict[str, Any], 
                                 user_id: str) -> Dict[str, Any]:
        """Prepare note record for database."""
        # Extract content from entities or use original text
        content = entities.get("content", result["original_text"])
        
        data = {
            "id": str(uuid.uuid4()),  # Generate UUID
            "user_id": user_id,
            "content": content,
            "source": "ai_pipeline",
            "created_at": datetime.now()
        }
        
        return {
            "table": "notes",
            "data": data
        }
    
    async def _prepare_ledger_record(self, 
                                   entities: Dict[str, Any], 
                                   result: Dict[str, Any], 
                                   user_id: str) -> Dict[str, Any]:
        """Prepare ledger record for database."""
        data = {
            "id": str(uuid.uuid4()),  # Generate UUID
            "user_id": user_id,
            "direction": "owe",  # Default direction 
            "created_at": datetime.now()
        }
        
        # Extract amount if available (handle both array and single value formats)
        amount_entities = entities.get("amount", [])
        if amount_entities:
            try:
                # Handle both array and single value formats
                if isinstance(amount_entities, list) and amount_entities:
                    amount_str = amount_entities[0]
                else:
                    amount_str = str(amount_entities)
                
                # Extract numeric value
                amount_match = re.search(r'(\d+(?:\.\d{2})?)', str(amount_str))
                if amount_match:
                    data["amount"] = float(amount_match.group(1))
                else:
                    data["amount"] = 0.0
            except (ValueError, IndexError, TypeError):
                data["amount"] = 0.0
        else:
            data["amount"] = 0.0
        
        # Extract contact name (handle both array and single value formats)
        person_entities = entities.get("person", [])
        contact_name = None
        
        if person_entities:
            try:
                # Handle both array and single value formats
                if isinstance(person_entities, list) and person_entities:
                    contact_name = person_entities[0]
                else:
                    contact_name = str(person_entities)
            except (TypeError, IndexError):
                contact_name = None
        
        # Fallback: extract contact name from original text if not found in entities
        if not contact_name:
            contact_name = self._extract_contact_name(result["original_text"], entities)
        
        if contact_name:
            data["contact_name"] = contact_name
        else:
            data["contact_name"] = "Unknown Contact"  # Default fallback
        
        return {
            "table": "ledger_entries",
            "data": data
        }
    
    def _extract_contact_name(self, original_text: str, entities: Dict[str, Any]) -> Optional[str]:
        """Extract contact name from text and entities."""
        try:
            text_lower = original_text.lower()
            
            # Look for common patterns like "Sarah owes me", "I owe John", etc.
            name_patterns = [
                r'(\w+)\s+owes?\s+me',  # "Sarah owes me"
                r'i\s+owe\s+(\w+)',     # "I owe John"
                r'lend\s+(\w+)',        # "lend John"
                r'borrow\s+from\s+(\w+)', # "borrow from Sarah"
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    name = match.group(1).capitalize()
                    # Skip common words
                    if name not in ['Me', 'You', 'I', 'He', 'She', 'They', 'That', 'Note', 'Money']:
                        return name
            
            # Fallback: look for capitalized words in the original text
            words = original_text.split()
            for word in words:
                # Look for capitalized words that could be names
                if (word[0].isupper() and len(word) > 2 and 
                    word.lower() not in ['note', 'that', 'owes', 'owe', 'money', 'dollars']):
                    return word
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract contact name from '{original_text}': {e}")
            return None
    
    async def _prepare_history_record(self, 
                                    entities: Dict[str, Any], 
                                    result: Dict[str, Any], 
                                    user_id: str) -> Dict[str, Any]:
        """Prepare history record for database."""
        data = {
            "id": str(uuid.uuid4()),  # Generate UUID
            "user_id": user_id,
            "content": result["original_text"],
            "interaction_type": "ai_chat",
            "created_at": datetime.now()
        }
        
        return {
            "table": "history_logs",
            "data": data
        }
    
    async def _insert_record(self, table_name: str, data: Dict[str, Any]) -> Optional[int]:
        """Insert record into specified table."""
        try:
            # Create a new database session
            session = SessionLocal()
            
            try:
                # Validate required fields
                schema = self.table_schemas.get(table_name)
                if not schema:
                    logger.error(f"Unknown table: {table_name}")
                    return None
                
                # Check required fields
                missing_fields = []
                for field in schema["required_fields"]:
                    if field not in data:
                        missing_fields.append(field)
                
                if missing_fields:
                    logger.error(f"Missing required fields for {table_name}: {missing_fields}")
                    return None
                
                # Add default values for missing optional fields
                for field, default_func in schema["default_values"].items():
                    if field not in data:
                        if callable(default_func):
                            data[field] = default_func()
                        else:
                            data[field] = default_func
                
                # Prepare the insert statement
                columns = ", ".join(data.keys())
                placeholders = ", ".join([f":{key}" for key in data.keys()])
                
                query = text(f"""
                    INSERT INTO {table_name} ({columns})
                    VALUES ({placeholders})
                    RETURNING id
                """)
                
                # Execute the insert
                result = session.execute(query, data)
                session.commit()
                
                # Get the inserted ID
                inserted_id = result.scalar()
                logger.info(f"Successfully inserted record into {table_name} with ID: {inserted_id}")
                
                return inserted_id
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Database error inserting into {table_name}: {e}")
                return None
            except Exception as e:
                session.rollback()
                logger.error(f"Unexpected error inserting into {table_name}: {e}")
                return None
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to create database session: {e}")
            return None
    
    async def validate_user_exists(self, user_id: str) -> bool:
        """Validate that the user exists in the database."""
        try:
            session = SessionLocal()
            try:
                query = text("SELECT id FROM users WHERE id = :user_id")
                result = session.execute(query, {"user_id": user_id})
                return result.scalar() is not None
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error validating user existence: {e}")
            return False
    
    async def create_user_if_not_exists(self, user_id: str) -> bool:
        """
        Ensure user exists in the system.
        
        Args:
            user_id: User ID to check/create
            
        Returns:
            True if user exists or was created successfully
        """
        try:
            # In this implementation, we assume users are managed by Firebase Auth
            # and don't need explicit creation in our database
            logger.info(f"User {user_id} verified for database operations")
            return True
            
        except Exception as e:
            logger.error(f"User verification failed for {user_id}: {e}")
            return False
    
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics across all tables."""
        try:
            session = SessionLocal()
            try:
                stats = {}
                
                # Count reminders
                query = text("SELECT COUNT(*) FROM reminders WHERE user_id = :user_id")
                result = session.execute(query, {"user_id": user_id})
                stats["total_reminders"] = result.scalar() or 0
                
                # Count notes
                query = text("SELECT COUNT(*) FROM notes WHERE user_id = :user_id")
                result = session.execute(query, {"user_id": user_id})
                stats["total_notes"] = result.scalar() or 0
                
                # Count ledger entries
                query = text("SELECT COUNT(*) FROM ledger_entries WHERE user_id = :user_id")
                result = session.execute(query, {"user_id": user_id})
                stats["total_ledger_entries"] = result.scalar() or 0
                
                # Count history logs
                query = text("SELECT COUNT(*) FROM history_logs WHERE user_id = :user_id")
                result = session.execute(query, {"user_id": user_id})
                stats["total_history_logs"] = result.scalar() or 0
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "statistics": stats
                }
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    async def get_recent_records(self, user_id: str, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent records for a user from a specific table."""
        try:
            if table_name not in self.table_schemas:
                return []
            
            session = SessionLocal()
            try:
                query = text(f"""
                    SELECT * FROM {table_name} 
                    WHERE user_id = :user_id 
                    ORDER BY created_at DESC 
                    LIMIT :limit
                """)
                
                result = session.execute(query, {"user_id": user_id, "limit": limit})
                rows = result.fetchall()
                
                return [dict(row._mapping) for row in rows]
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting recent records from {table_name}: {e}")
            return []
    
    def get_supported_tables(self) -> List[str]:
        """Get list of supported database tables."""
        return list(self.table_schemas.keys())
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get schema information for a specific table."""
        return self.table_schemas.get(table_name)

# Export the service
__all__ = ["DatabaseIntegrationService"] 