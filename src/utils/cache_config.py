"""
Cache Configuration Utility

This module provides utilities to ensure all caching operations use centralized paths.
Especially useful for Jupyter notebooks to maintain consistency with the main application.
"""

import os
from pathlib import Path


def setup_cache_paths(cache_dir: str = "./cache", model_cache_dir: str = "./models") -> dict:
    """
    Configure HuggingFace environment variables to use centralized cache paths.
    
    This ensures all HuggingFace operations (datasets, models, tokenizers) use
    the specified cache directories. Critical for cloud deployments where caches
    should be on persistent volumes.
    
    Args:
        cache_dir: Directory for HuggingFace datasets cache
        model_cache_dir: Directory for HuggingFace models/transformers cache
    
    Returns:
        Dictionary with absolute cache paths for reference
        
    Example:
        >>> # In a notebook:
        >>> from src.utils.cache_config import setup_cache_paths
        >>> cache_paths = setup_cache_paths(
        ...     cache_dir="/mnt/persistent/cache",
        ...     model_cache_dir="/mnt/persistent/models"
        ... )
        >>> print(f"Using cache at: {cache_paths['datasets']}")
    """
    # Convert to absolute paths
    abs_cache_dir = str(Path(cache_dir).resolve())
    abs_model_cache_dir = str(Path(model_cache_dir).resolve())
    
    # Create directories if they don't exist
    Path(abs_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(abs_model_cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Set HuggingFace environment variables
    os.environ['HF_DATASETS_CACHE'] = abs_cache_dir
    os.environ['HF_HOME'] = abs_model_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = abs_model_cache_dir
    os.environ['HF_HUB_CACHE'] = abs_model_cache_dir
    
    return {
        'datasets': abs_cache_dir,
        'models': abs_model_cache_dir,
        'env_vars': {
            'HF_DATASETS_CACHE': abs_cache_dir,
            'HF_HOME': abs_model_cache_dir,
            'TRANSFORMERS_CACHE': abs_model_cache_dir,
            'HF_HUB_CACHE': abs_model_cache_dir
        }
    }


def get_cache_paths() -> dict:
    """
    Get currently configured cache paths from environment variables.
    
    Returns:
        Dictionary with current cache paths
    """
    return {
        'datasets': os.environ.get('HF_DATASETS_CACHE', 'Not configured'),
        'models': os.environ.get('HF_HOME', 'Not configured'),
        'transformers': os.environ.get('TRANSFORMERS_CACHE', 'Not configured'),
        'hub': os.environ.get('HF_HUB_CACHE', 'Not configured')
    }
