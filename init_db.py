from connect_db import init_db
from models.models import User, Note, Reminder, LedgerEntry, Friendship
from utils.logger import logger

if __name__ == "__main__":
    try:
        init_db()
        logger.info("Database initialized successfully with all models")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise 