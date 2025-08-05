from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import settings
import logging

# Import all models to ensure they are registered with Base
from .models import Base, Customer, Friendship, FriendPermission, FriendRequestHistory

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    echo=settings.DEBUG
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    try:
        logger.info(f"Attempting to create tables for: {list(Base.metadata.tables.keys())}")
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        logger.info(f"Tables found in database: {existing_tables}")
        
        expected_tables = list(Base.metadata.tables.keys())
        missing_tables = [table for table in expected_tables if table not in existing_tables]
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
        else:
            logger.info("All expected tables are present")
            
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise
