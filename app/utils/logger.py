import logging
import sys
from app.config import settings

def setup_logging():
    """Setup application logging"""
    
    # Set log level based on debug setting
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )