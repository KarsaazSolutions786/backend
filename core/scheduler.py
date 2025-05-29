from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from datetime import datetime
from utils.logger import logger
from connect_db import DATABASE_URL

# Configure scheduler with database job store
jobstores = {
    'default': SQLAlchemyJobStore(url=DATABASE_URL)
}

executors = {
    'default': AsyncIOExecutor()
}

job_defaults = {
    'coalesce': False,
    'max_instances': 3
}

scheduler = AsyncIOScheduler(
    jobstores=jobstores,
    executors=executors,
    job_defaults=job_defaults,
    timezone='UTC'
)

async def send_reminder_notification(reminder_id: str, user_id: str, message: str):
    """Send a reminder notification to the user."""
    logger.info(f"Sending reminder notification: {reminder_id} to user {user_id}")
    
    # Here you would implement actual notification logic
    # For now, just log the reminder
    print(f"🔔 REMINDER: {message} (User: {user_id})")
    
    # You could integrate with:
    # - Push notifications
    # - Email service
    # - SMS service
    # - WebSocket for real-time notifications

def schedule_reminder(reminder_id: str, user_id: str, message: str, scheduled_time: datetime):
    """Schedule a reminder notification."""
    try:
        scheduler.add_job(
            send_reminder_notification,
            'date',
            run_date=scheduled_time,
            args=[reminder_id, user_id, message],
            id=f"reminder_{reminder_id}",
            replace_existing=True
        )
        logger.info(f"Scheduled reminder {reminder_id} for {scheduled_time}")
        return True
    except Exception as e:
        logger.error(f"Failed to schedule reminder {reminder_id}: {e}")
        return False

def cancel_reminder(reminder_id: str):
    """Cancel a scheduled reminder."""
    try:
        scheduler.remove_job(f"reminder_{reminder_id}")
        logger.info(f"Cancelled reminder {reminder_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel reminder {reminder_id}: {e}")
        return False

def reschedule_reminder(reminder_id: str, user_id: str, message: str, new_time: datetime):
    """Reschedule an existing reminder."""
    cancel_reminder(reminder_id)
    return schedule_reminder(reminder_id, user_id, message, new_time) 