from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import logging

from ..models import (
    Reminder, Customer, PriorityLevel, Timezone,
    ReminderNotification, ReminderShare
)

logger = logging.getLogger(__name__)

class ReminderService:
    """Reminder service with real database operations"""
    
    def __init__(self):
        """Initialize the service"""
        pass
    
    def get_priority_mapping(self):
        """Map API priority strings to database labels"""
        return {
            "low": "Low",
            "medium": "Medium", 
            "high": "High",
            "urgent": "High"  # Map urgent to High since only Low/Medium/High exist
        }
    
    def get_reverse_priority_mapping(self):
        """Map database labels to API priority strings"""
        return {
            "Low": "low",
            "Medium": "medium",
            "High": "high"
        }
    
    async def ensure_customer_exists(self, db: Session, customer_id: str) -> Customer:
        """Ensure customer record exists for the customer_id"""
        try:
            # Convert customer_id to integer (auth service returns string, we need int for DB)
            customer_id_int = int(customer_id)
            
            customer = db.query(Customer).filter(Customer.id == customer_id_int).first()
            if not customer:
                # Create a minimal customer record
                customer = Customer(
                    id=customer_id_int,
                    email=f"user{customer_id_int}@eindr.com",  # Placeholder email
                    is_active=True
                )
                db.add(customer)
                db.commit()
                logger.info(f"Created customer record for customer_id: {customer_id}")
                
            return customer
            
        except ValueError:
            logger.error(f"Invalid customer_id format: {customer_id}")
            raise ValueError(f"Invalid customer_id format: {customer_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error ensuring customer exists: {e}")
            raise

    async def create_reminder(self, db: Session, customer_id: str, reminder_data: dict):
        """Create a new reminder with proper foreign key lookups"""
        try:
            # Ensure customer exists
            customer = await self.ensure_customer_exists(db, customer_id)
            
            # Look up priority_id using label field
            priority_id = None
            if reminder_data.get("priority"):
                priority_mapping = self.get_priority_mapping()
                priority_label = priority_mapping.get(reminder_data["priority"])
                if priority_label:
                    priority = db.query(PriorityLevel).filter(
                        PriorityLevel.label == priority_label
                    ).first()
                    if priority:
                        priority_id = priority.id
            
            # Look up timezone_id using name field  
            timezone_id = None
            if reminder_data.get("timezone"):
                timezone = db.query(Timezone).filter(
                    Timezone.name == reminder_data["timezone"]
                ).first()
                if timezone:
                    timezone_id = timezone.id
            
            # Handle repeat_pattern as simple string mapping to ID
            # Since no repeat_pattern table exists, we'll use simple numeric mapping
            repeat_pattern_id = None
            if reminder_data.get("repeat_pattern"):
                repeat_mapping = {
                    "none": 0,
                    "daily": 1,
                    "weekly": 2,
                    "monthly": 3,
                    "yearly": 4
                }
                repeat_pattern_id = repeat_mapping.get(reminder_data["repeat_pattern"], 0)
            
            # Create reminder record
            reminder = Reminder(
                customer_id=customer.id,
                title=reminder_data.get("title"),
                description=reminder_data.get("description"),
                time=reminder_data.get("time"),
                priority_id=priority_id,
                repeat_pattern_id=repeat_pattern_id,
                timezone_id=timezone_id,
                is_active=True,
                is_completed=False,
                occurrence_count=0
            )
            
            db.add(reminder)
            db.commit()
            db.refresh(reminder)
            
            logger.info(f"Created reminder {reminder.id} for customer {customer.id}")
            return reminder
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating reminder: {e}")
            raise

    async def get_user_reminders(self, db: Session, customer_id: str, skip: int, limit: int, filters: dict):
        """Get reminders for a customer with filtering"""
        try:
            customer_id_int = int(customer_id)
            
            query = db.query(Reminder).filter(
                Reminder.customer_id == customer_id_int,
                Reminder.is_active == True
            )
            
            # Apply filters
            if filters.get("priority"):
                priority_mapping = self.get_priority_mapping()
                priority_label = priority_mapping.get(filters["priority"])
                if priority_label:
                    priority = db.query(PriorityLevel).filter(
                        PriorityLevel.label == priority_label
                    ).first()
                    if priority:
                        query = query.filter(Reminder.priority_id == priority.id)
            
            if filters.get("status"):
                if filters["status"] == "completed":
                    query = query.filter(Reminder.is_completed == True)
                elif filters["status"] == "pending":
                    query = query.filter(Reminder.is_completed == False)
            
            if filters.get("from_date"):
                query = query.filter(Reminder.time >= filters["from_date"])
            
            if filters.get("to_date"):
                query = query.filter(Reminder.time <= filters["to_date"])
            
            return query.offset(skip).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Error getting user reminders: {e}")
            return []

    async def get_reminder(self, db: Session, reminder_id: str, customer_id: str):
        """Get a specific reminder"""
        try:
            customer_id_int = int(customer_id)
            reminder_id_int = int(reminder_id)
            
            reminder = db.query(Reminder).filter(
                Reminder.id == reminder_id_int,
                Reminder.customer_id == customer_id_int,
                Reminder.is_active == True
            ).first()
            
            return reminder
            
        except Exception as e:
            logger.error(f"Error getting reminder: {e}")
            return None

    async def update_reminder(self, db: Session, reminder_id: str, customer_id: str, update_data: dict):
        """Update a reminder"""
        try:
            reminder = await self.get_reminder(db, reminder_id, customer_id)
            if not reminder:
                return None
            
            # Update fields
            for key, value in update_data.items():
                if key == "priority" and value:
                    priority_mapping = self.get_priority_mapping()
                    priority_label = priority_mapping.get(value)
                    if priority_label:
                        priority = db.query(PriorityLevel).filter(
                            PriorityLevel.label == priority_label
                        ).first()
                        if priority:
                            reminder.priority_id = priority.id
                elif key == "repeat_pattern" and value:
                    repeat_mapping = {
                        "none": 0,
                        "daily": 1,
                        "weekly": 2,
                        "monthly": 3,
                        "yearly": 4
                    }
                    reminder.repeat_pattern_id = repeat_mapping.get(value, 0)
                elif key == "timezone" and value:
                    timezone = db.query(Timezone).filter(
                        Timezone.name == value
                    ).first()
                    if timezone:
                        reminder.timezone_id = timezone.id
                elif hasattr(reminder, key):
                    setattr(reminder, key, value)
            
            reminder.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(reminder)
            
            return reminder
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating reminder: {e}")
            return None

    async def delete_reminder(self, db: Session, reminder_id: str, customer_id: str):
        """Soft delete a reminder"""
        try:
            reminder = await self.get_reminder(db, reminder_id, customer_id)
            if not reminder:
                return False
            
            reminder.is_active = False
            reminder.updated_at = datetime.utcnow()
            db.commit()
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting reminder: {e}")
            return False

    async def complete_reminder(self, db: Session, reminder_id: str, customer_id: str):
        """Mark a reminder as completed"""
        try:
            reminder = await self.get_reminder(db, reminder_id, customer_id)
            if not reminder:
                return None
            
            reminder.is_completed = True
            reminder.completed_at = datetime.utcnow()
            reminder.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(reminder)
            
            return reminder
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error completing reminder: {e}")
            return None

    async def snooze_reminder(self, db: Session, reminder_id: str, customer_id: str, snooze_minutes: int):
        """Snooze a reminder"""
        try:
            reminder = await self.get_reminder(db, reminder_id, customer_id)
            if not reminder:
                return None
            
            # Add snooze minutes to the current time
            new_time = reminder.time + timedelta(minutes=snooze_minutes)
            reminder.time = new_time
            reminder.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(reminder)
            
            return reminder
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error snoozing reminder: {e}")
            return None

    async def share_reminder(self, db: Session, reminder_id: str, owner_customer_id: str, shared_with_customer_id: str, permissions: dict):
        """Share a reminder with another customer"""
        try:
            # This is a stub for now - would need more complex customer lookup
            return None
            
        except Exception as e:
            logger.error(f"Error sharing reminder: {e}")
            return None

    async def get_shared_with_user(self, db: Session, customer_id: str, skip: int, limit: int):
        """Get reminders shared with customer"""
        # Stub for now
        return []

    async def get_reminders_in_timeframe(self, db: Session, customer_id: str, from_time: datetime, to_time: datetime):
        """Get reminders in a specific timeframe"""
        try:
            customer_id_int = int(customer_id)
            
            reminders = db.query(Reminder).filter(
                Reminder.customer_id == customer_id_int,
                Reminder.is_active == True,
                Reminder.is_completed == False,
                Reminder.time >= from_time,
                Reminder.time <= to_time
            ).all()
            
            return reminders
            
        except Exception as e:
            logger.error(f"Error getting reminders in timeframe: {e}")
            return [] 