"""
Centralized Configuration Management for Whisper ASR Fine-tuning.
This module provides a comprehensive configuration system for all constants and parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    # Dataset source
    # dataset_name: str = "SPEAK-ASR/openslr-sinhala-asr"
    # train_split: str = "train[:21000]+validation[:3000]"
    # test_split: str = "test[:6000]"
    dataset_name: str = "SPEAK-ASR/youtube-sinhala-asr"
    train_split: str = "train+validation"
    test_split: str = "test"
    use_auth_token: bool = True

    # Audio preprocessing
    sample_rate: int = 16000
    audio_column: str = "audio"
    transcript_column: str = "text"


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    
    # Model selection
    model_name: str = "openai/whisper-small"
    language: str = "Sinhala"
    task: str = "transcribe"  # "transcribe" or "translate"
    max_token_length: int = 1024


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training identification
    run_name: str = "whisper-sinhala-finetune"
    output_dir: str = "./whisper-small-sinhala"
    
    # Training epochs/steps
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # Batch sizes
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    auto_find_batch_size: bool = True
    
    # Learning rate
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    lr_scheduler_type: str = "linear"

    # Optimization
    gradient_checkpointing: bool = True
    use_cache: bool = False
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_torch"
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 250
    predict_with_generate: bool = True
    generation_max_length: int = None
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 250
    load_best_model_at_end: bool = True
    
    # Metrics
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 25
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["mlflow"])
    
    # Hub integration
    push_to_hub: bool = True
    hub_strategy: str = "every_save"
    
    # Advanced features
    neftune_noise_alpha: Optional[float] = 5.0  # 5.0-15.0 for NEFTune, None to disable
    weight_decay: float = 0.01


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "SEQ_2_SEQ_LM"


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace Hub integration."""
    
    # Authentication
    use_auth_token: bool = True
    hf_token_env_var: str = "HF_TOKEN"
    pretty_name: str = "Whisper Small - Sinhala ASR Fine-Tuned"
    dataset_args: str = "config: si, split: test"
    model_name: str = "speak-whisper-small-si"
    hub_repo_id: str = "SPEAK-ASR/speak-whisper-small-si"
    tasks: str = "automatic-speech-recognition"


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories
    project_root: str = field(default_factory=lambda: os.getcwd())
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    
    # Model directories
    model_cache_dir: str = "./models"
    checkpoint_dir: str = "./checkpoints"
    
    # Log directories
    log_dir: str = "./logs"
    tensorboard_dir: str = "./runs"


@dataclass
class Config:
    """
    Master configuration class that aggregates all configuration sections.
    This is the main configuration object to use throughout the application.
    """
    
    # Configuration sections
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    

# Configuration instance for convenience
CONFIG = Config()