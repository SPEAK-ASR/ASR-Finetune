"""
Logging utility for the Whisper fine-tuning project.
Provides consistent logging across all modules.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional

from src.config.config import CONFIG


def _ensure_log_dir() -> str:
    """Ensure the log directory exists and return its path."""
    log_dir = CONFIG.paths.log_dir
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _get_log_file_path() -> str:
    """Generate a log file path with timestamp."""
    log_dir = _ensure_log_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"asr_finetune_{timestamp}.log")


# Global log file path (created once per session)
_SESSION_LOG_FILE: Optional[str] = None


def _get_session_log_file() -> str:
    """Get or create the session log file path."""
    global _SESSION_LOG_FILE
    if _SESSION_LOG_FILE is None:
        _SESSION_LOG_FILE = _get_log_file_path()
    return _SESSION_LOG_FILE


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file output.
    
    Args:
        name: Name of the logger (typically __name__ of the module)
        level: Logging level (default: INFO)
        log_file: Optional custom path to log file (uses session log file if not provided)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Format: timestamp - logger name - level - message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - always add to log file
    log_file_path = log_file if log_file else _get_session_log_file()
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
