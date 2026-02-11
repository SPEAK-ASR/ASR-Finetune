"""
Centralized Configuration Management for Whisper ASR Fine-tuning.
This module provides a comprehensive configuration system for all constants and parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from src.config.training import TrainingConfig, get_training_config
from src.config.huggingface import HuggingFaceConfig, get_huggingface_config
from src.config.lora import LoRAConfig, get_lora_config
from src.config.dataset import DatasetConfig, get_dataset_config
from src.config.model import ModelConfig, get_model_config


@dataclass
class RuntimeConfig:
    """Configuration for runtime/execution settings."""
    
    # task can be prepare_dataset, finetune_asr_model, or optuna_optimize
    task: Literal["prepare_dataset", "finetune_asr_model", "optuna_optimize"] = "optuna_optimize"
    
    # Checkpoint resumption
    resume_from_checkpoint: Optional[Union[str, bool]] = False
    early_stopping_patience: int = 3
    
    # Optuna hyperparameter optimization settings
    optuna_n_trials: int = 50
    optuna_trial_epochs: float = 1.0


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Cache directories
    cache_dir: str = "./cache"  # HuggingFace datasets cache
    model_cache_dir: str = "./models"  # HuggingFace models cache
    
    # Log directories
    log_dir: str = "./logs"  # Application logs
    wandb_dir: str = "./wandb"  # Weights & Biases logs
    
    def __post_init__(self):
        """Initialize paths and set environment variables for HuggingFace caching."""
        # Create directories if they don't exist
        self._ensure_directories_exist()
        
        # Set HuggingFace environment variables to use our centralized cache paths
        self._configure_huggingface_cache()
    
    def _ensure_directories_exist(self) -> None:
        """Create all configured directories if they don't exist."""
        directories = [
            self.cache_dir,
            self.model_cache_dir,
            self.log_dir,
            self.wandb_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _configure_huggingface_cache(self) -> None:
        """Set HuggingFace environment variables to respect our cache configuration.
        
        This ensures that all HuggingFace operations (datasets, models, tokenizers)
        use our centralized cache paths, even if not explicitly passed as parameters.
        This is crucial for cloud deployments with persistent volumes.
        """
        # Convert to absolute paths for reliability
        abs_cache_dir = str(Path(self.cache_dir).resolve())
        abs_model_cache_dir = str(Path(self.model_cache_dir).resolve())
        
        # HuggingFace Datasets cache
        os.environ['HF_DATASETS_CACHE'] = abs_cache_dir
        
        # HuggingFace Hub cache (models, tokenizers, configs)
        os.environ['HF_HOME'] = abs_model_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = abs_model_cache_dir
        os.environ['HF_HUB_CACHE'] = abs_model_cache_dir


@dataclass
class Config:
    """
    Master configuration class that aggregates all configuration sections.
    This is the main configuration object to use throughout the application.
    """
    
    # Configuration sections
    dataset: DatasetConfig = field(default_factory=get_dataset_config)
    model: ModelConfig = field(default_factory=get_model_config)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=get_training_config)
    lora: LoRAConfig = field(default_factory=get_lora_config)
    huggingface: HuggingFaceConfig = field(default_factory=get_huggingface_config)
    paths: PathConfig = field(default_factory=PathConfig)
    

# Configuration instance for convenience
CONFIG = Config()