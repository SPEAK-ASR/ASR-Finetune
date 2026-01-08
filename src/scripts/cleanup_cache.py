"""
Utility script to clean up HuggingFace cache files.
"""
import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def cleanup_cache():
    """Remove all HuggingFace dataset cache files."""
    cache_dir = config.HF_DATASETS_CACHE
    
    logger.info(f"Cache directory: {cache_dir}")
    
    try:
        if os.path.exists(cache_dir):
            # Get size before
            size_before = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(cache_dir)
                for filename in filenames
            ) / (1024 ** 3)
            
            logger.info(f"Cache size before: {size_before:.2f} GB")
            
            # Remove cache
            shutil.rmtree(cache_dir)
            logger.info("✓ Cache cleared successfully")
        else:
            logger.info("No cache directory found")
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")

if __name__ == "__main__":
    cleanup_cache()