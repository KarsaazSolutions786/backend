#!/usr/bin/env python3
"""
Database initialization fallback script
This script ensures required columns exist even if migrations fail
"""

import os
import sys
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError

def check_and_add_columns():
    """Check and add missing columns to ledger_entries table"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("No DATABASE_URL found, skipping column check")
        return True
    
    try:
        engine = create_engine(database_url)
        inspector = inspect(engine)
        
        # Check if ledger_entries table exists
        if 'ledger_entries' not in inspector.get_table_names():
            print("ledger_entries table doesn't exist, skipping column check")
            return True
        
        # Get existing columns
        existing_columns = [col['name'] for col in inspector.get_columns('ledger_entries')]
        print(f"Existing columns in ledger_entries: {existing_columns}")
        
        # Define required columns
        required_columns = {
            'friend_name': 'VARCHAR(255)',
            'friend_phone': 'VARCHAR(20)',
            'friend_email': 'VARCHAR(255)'
        }
        
        # Add missing columns
        with engine.connect() as conn:
            for column_name, column_type in required_columns.items():
                if column_name not in existing_columns:
                    try:
                        sql = f"ALTER TABLE ledger_entries ADD COLUMN {column_name} {column_type}"
                        conn.execute(text(sql))
                        conn.commit()
                        print(f"Added missing column: {column_name}")
                    except SQLAlchemyError as e:
                        print(f"Failed to add column {column_name}: {e}")
                        # Continue with other columns
                else:
                    print(f"Column {column_name} already exists")
        
        print("Database column check completed")
        return True
        
    except Exception as e:
        print(f"Database column check failed: {e}")
        return False

if __name__ == "__main__":
    success = check_and_add_columns()
    sys.exit(0 if success else 1)