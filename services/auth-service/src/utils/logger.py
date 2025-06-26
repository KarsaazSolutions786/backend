import logging
import sys
from pathlib import Path

def setup_logger(name: str = None, level: str = "INFO") -> logging.Logger:
    """Setup logger with consistent formatting"""
    
    # Create logger
    logger = logging.getLogger(name or __name__)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Set level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

# Create default logger
logger = setup_logger("auth-service") 