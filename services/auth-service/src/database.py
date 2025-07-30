from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings
import logging

# Import RefreshToken model from shared module
import sys
import os

try:
    from shared.refresh_token_service import RefreshTokenBase
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import RefreshTokenBase from shared module: {e}")
    RefreshTokenBase = None

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

# Create Base class
Base = declarative_base()

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
        # Create auth-service tables
        Base.metadata.create_all(bind=engine)
        
        # Create refresh token tables if available
        if RefreshTokenBase is not None:
            RefreshTokenBase.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully (including refresh tokens)")
        else:
            logger.warning("RefreshTokenBase not available - refresh token table not created")
            logger.info("Database tables created successfully (auth tables only)")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise