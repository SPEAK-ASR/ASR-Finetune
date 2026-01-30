"""
Centralized Configuration Management for Whisper ASR Fine-tuning.
This module provides a comprehensive configuration system for all constants and parameters.
"""

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
    
    # task can be either prepare_dataset or finetune_asr_model
    task: Literal["prepare_dataset", "finetune_asr_model"] = "finetune_asr_model"
    
    # Checkpoint resumption
    resume_from_checkpoint: Optional[Union[str, bool]] = False
    early_stopping_patience: int = 3


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Cache directories
    cache_dir: str = "./cache"  # HuggingFace datasets cache
    model_cache_dir: str = "./models"  # HuggingFace models cache
    
    # Log directories
    log_dir: str = "./logs"  # Application logs
    wandb_dir: str = "./wandb"  # Weights & Biases logs


@dataclass
class Config:
    """
    Master configuration class that aggregates all configuration sections.
    This is the main configuration object to use throughout the application.
    """
    
    # Configuration sections
    dataset: DatasetConfig = get_dataset_config()
    model: ModelConfig = get_model_config()
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = get_training_config()
    lora: LoRAConfig = get_lora_config()
    huggingface: HuggingFaceConfig = get_huggingface_config()
    paths: PathConfig = field(default_factory=PathConfig)
    

# Configuration instance for convenience
CONFIG = Config()