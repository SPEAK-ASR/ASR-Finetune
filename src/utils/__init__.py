"""
Utility modules for the Whisper ASR fine-tuning project.
"""

from .logger import setup_logger
from .cache_config import setup_cache_paths, get_cache_paths

__all__ = ["setup_logger", "setup_cache_paths", "get_cache_paths"]
