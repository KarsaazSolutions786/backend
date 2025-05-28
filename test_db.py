#!/usr/bin/env python3
"""Test database connection and create tables."""

import os
from connect_db import engine, Base, DATABASE_URL
from models.models import User, Preferences, Reminder, Note, LedgerEntry, Friendship, Permission, Embedding, HistoryLog
from sqlalchemy import text

def main():
    print(f"Database URL: {DATABASE_URL}")
    print(f"Environment DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
    
    try:
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Database connection successful!")
            print(f"PostgreSQL version: {version}")
        
        # Create tables
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")
        
        # List tables
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = result.fetchall()
            print(f"\nTables in database:")
            for table in tables:
                print(f"  - {table[0]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 