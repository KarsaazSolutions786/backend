#!/usr/bin/env python3
"""
Railway Database Initialization Script

This script initializes the database for Railway deployment.
It creates all necessary tables and can be run as a one-off command.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from connect_db import create_tables, engine
from utils.logger import logger
import models.models  # Import all models to register them


def init_railway_db():
    """Initialize database for Railway deployment."""
    try:
        logger.info("Starting Railway database initialization...")
        
        # Log database URL (without credentials)
        db_url = os.getenv("DATABASE_URL", "Not set")
        logger.info(f"Using database: {db_url.split('@')[1] if '@' in db_url else 'local'}")
        
        # Create all tables
        create_tables()
        
        logger.info("✅ Database initialized successfully for Railway!")
        logger.info("All tables created and ready for use.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = init_railway_db()
    sys.exit(0 if success else 1) 