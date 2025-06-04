"""
Database Integration Service for Intent Classification Results
Handles storing classification results into appropriate PostgreSQL tables.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import text, select, insert, update
    from sqlalchemy.exc import SQLAlchemyError
    
    # Try different possible database imports
    try:
        from core.database import get_db_session
    except ImportError:
        try:
            from connect_db import get_db_session
        except ImportError:
            from connect_db import SessionLocal
            # Create a fallback async session function
            async def get_db_session():
                """Fallback database session for sync databases."""
                db = SessionLocal()
                try:
                    yield db
                finally:
                    db.close()
    
    DATABASE_AVAILABLE = True
except ImportError as e:
    # Fallback when database dependencies are not available
    DATABASE_AVAILABLE = False
    
    # Mock classes for testing
    class AsyncSession:
        pass
    
    class SQLAlchemyError(Exception):
        pass
    
    def text(query):
        return query
    
    async def get_db_session():
        """Mock database session for testing."""
        return None

from utils.logger import logger

class DatabaseIntegrationService:
    """
    Service to integrate intent classification results with database storage.
    Handles both single and multi-intent classifications.
    """
    
    def __init__(self):
        self.table_schemas = {
            "reminders": {
                "required_fields": ["title", "user_id"],
                "optional_fields": ["time", "status", "description", "created_at"],
                "default_values": {"status": "active", "created_at": datetime.now}
            },
            "notes": {
                "required_fields": ["content", "user_id"],
                "optional_fields": ["category", "tags", "created_at"],
                "default_values": {"category": "general", "created_at": datetime.now}
            },
            "ledger_entries": {
                "required_fields": ["user_id"],
                "optional_fields": ["amount", "contact_name", "transaction_type", "description", "created_at"],
                "default_values": {"amount": 0.0, "transaction_type": "general", "created_at": datetime.now}
            },
            "history_logs": {
                "required_fields": ["message", "user_id"],
                "optional_fields": ["response_type", "created_at", "metadata"],
                "default_values": {"response_type": "general", "created_at": datetime.now}
            }
        }
    
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
    
    async def _prepare_reminder_record(self, 
                                     entities: Dict[str, Any], 
                                     result: Dict[str, Any], 
                                     user_id: str) -> Dict[str, Any]:
        """Prepare reminder record for database."""
        data = {
            "user_id": user_id,
            "title": entities.get("title", result["original_text"]),
            "status": "active",
            "created_at": datetime.now()
        }
        
        # Add time if available
        if entities.get("time"):
            data["time"] = entities["time"]
        
        # Add description with classification metadata
        data["description"] = json.dumps({
            "original_text": result["original_text"],
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used", "unknown"),
            "entities": entities
        })
        
        return {
            "table": "reminders",
            "data": data
        }
    
    async def _prepare_note_record(self, 
                                 entities: Dict[str, Any], 
                                 result: Dict[str, Any], 
                                 user_id: str) -> Dict[str, Any]:
        """Prepare note record for database."""
        data = {
            "user_id": user_id,
            "content": entities.get("content", result["original_text"]),
            "category": "general",
            "created_at": datetime.now()
        }
        
        # Determine category from content
        content_lower = data["content"].lower()
        if any(word in content_lower for word in ["shopping", "grocery", "buy", "purchase"]):
            data["category"] = "shopping"
        elif any(word in content_lower for word in ["meeting", "work", "project", "deadline"]):
            data["category"] = "work"
        elif any(word in content_lower for word in ["idea", "thought", "brainstorm"]):
            data["category"] = "ideas"
        
        # Add tags as JSON metadata
        data["tags"] = json.dumps({
            "original_text": result["original_text"],
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used", "unknown"),
            "entities": entities,
            "auto_categorized": True
        })
        
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
            "user_id": user_id,
            "amount": entities.get("amount", 0.0),
            "transaction_type": entities.get("transaction_type", "general"),
            "description": result["original_text"],
            "created_at": datetime.now()
        }
        
        # Add contact name if available
        if entities.get("contact_name"):
            data["contact_name"] = entities["contact_name"]
        
        # Add metadata
        data["metadata"] = json.dumps({
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used", "unknown"),
            "entities": entities,
            "auto_classified": True
        })
        
        return {
            "table": "ledger_entries",
            "data": data
        }
    
    async def _prepare_history_record(self, 
                                    entities: Dict[str, Any], 
                                    result: Dict[str, Any], 
                                    user_id: str) -> Dict[str, Any]:
        """Prepare history record for database."""
        data = {
            "user_id": user_id,
            "message": result["original_text"],
            "response_type": "chit_chat",
            "created_at": datetime.now()
        }
        
        # Add metadata
        data["metadata"] = json.dumps({
            "intent": result.get("intent", "chit_chat"),
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used", "unknown"),
            "entities": entities
        })
        
        return {
            "table": "history_logs",
            "data": data
        }
    
    async def _insert_record(self, table_name: str, data: Dict[str, Any]) -> Optional[int]:
        """Insert record into specified table."""
        try:
            async with get_db_session() as session:
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
                result = await session.execute(query, data)
                await session.commit()
                
                # Get the inserted ID
                inserted_id = result.scalar()
                logger.info(f"Successfully inserted record into {table_name} with ID: {inserted_id}")
                
                return inserted_id
                
        except SQLAlchemyError as e:
            logger.error(f"Database error inserting into {table_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error inserting into {table_name}: {e}")
            return None
    
    async def validate_user_exists(self, user_id: str) -> bool:
        """Validate that the user exists in the database."""
        try:
            async with get_db_session() as session:
                query = text("SELECT id FROM users WHERE id = :user_id")
                result = await session.execute(query, {"user_id": user_id})
                return result.scalar() is not None
                
        except Exception as e:
            logger.error(f"Error validating user existence: {e}")
            return False
    
    async def create_user_if_not_exists(self, user_id: str, user_data: Dict[str, Any] = None) -> bool:
        """Create user if they don't exist."""
        try:
            if await self.validate_user_exists(user_id):
                return True
            
            # Create new user
            async with get_db_session() as session:
                user_data = user_data or {
                    "id": user_id,
                    "created_at": datetime.now(),
                    "is_active": True
                }
                
                query = text("""
                    INSERT INTO users (id, created_at, is_active)
                    VALUES (:id, :created_at, :is_active)
                    ON CONFLICT (id) DO NOTHING
                """)
                
                await session.execute(query, user_data)
                await session.commit()
                
                logger.info(f"Created user: {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
    
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics across all tables."""
        try:
            async with get_db_session() as session:
                stats = {}
                
                # Count reminders
                query = text("SELECT COUNT(*) FROM reminders WHERE user_id = :user_id")
                result = await session.execute(query, {"user_id": user_id})
                stats["total_reminders"] = result.scalar() or 0
                
                # Count notes
                query = text("SELECT COUNT(*) FROM notes WHERE user_id = :user_id")
                result = await session.execute(query, {"user_id": user_id})
                stats["total_notes"] = result.scalar() or 0
                
                # Count ledger entries
                query = text("SELECT COUNT(*) FROM ledger_entries WHERE user_id = :user_id")
                result = await session.execute(query, {"user_id": user_id})
                stats["total_ledger_entries"] = result.scalar() or 0
                
                # Count history logs
                query = text("SELECT COUNT(*) FROM history_logs WHERE user_id = :user_id")
                result = await session.execute(query, {"user_id": user_id})
                stats["total_history_logs"] = result.scalar() or 0
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "statistics": stats
                }
                
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
            
            async with get_db_session() as session:
                query = text(f"""
                    SELECT * FROM {table_name} 
                    WHERE user_id = :user_id 
                    ORDER BY created_at DESC 
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {"user_id": user_id, "limit": limit})
                rows = result.fetchall()
                
                return [dict(row._mapping) for row in rows]
                
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