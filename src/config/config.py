"""
Centralized Configuration Management for Whisper ASR Fine-tuning.
This module provides a comprehensive configuration system for all constants and parameters.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Union
import os

HF_MODEL_ID = "SPEAK-ASR/speak-whisper-small-si-full-dataset"

@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset source."""
    dataset_name: str
    train_split: str | None = "train"
    test_split: str | None = "test"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    # task can be either prepare_dataset or finetune_asr_model
    task: Literal["prepare_dataset", "finetune_asr_model"] = "finetune_asr_model"
    
    # Multiple datasets configuration
    # Each dataset has: dataset_name, train_split, test_split
    # All datasets will be combined into a single DatasetDict
    datasets: List[SingleDatasetConfig] = field(default_factory=lambda: [
        SingleDatasetConfig(
            dataset_name="SPEAK-ASR/openslr-sinhala-asr-preprocessed-1",
            train_split="train",
            test_split="test"
        ),
        SingleDatasetConfig(
            dataset_name="SPEAK-ASR/openslr-sinhala-asr-preprocessed-2",
            train_split="train",
            test_split="test"
        ),
        SingleDatasetConfig(
            dataset_name="SPEAK-ASR/openslr-sinhala-asr-preprocessed-3",
            train_split="train",
            test_split=None
        )
    ])

    use_auth_token: bool = True
    keep_in_memory: bool = False  # Whether to load datasets into memory

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
class RuntimeConfig:
    """Configuration for runtime/execution settings."""
    
    # Checkpoint resumption
    resume_from_checkpoint: Optional[Union[str, bool]] = False  # Path to checkpoint or False


@dataclass
class TrainingConfig:
    """Configuration for model training (Seq2SeqTrainingArguments compatible)."""
    
    # Training identification
    # run_name: str = "whisper-sinhala-finetune"
    output_dir: str = "checkpoints"
    
    # Training epochs/steps
    num_train_epochs: int = 3
    # max_steps: int = -1
    
    # Batch sizes
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    auto_find_batch_size: bool = True
    
    # Learning rate
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    # lr_scheduler_type: str = "linear"

    # Optimization
    # gradient_checkpointing: bool = True
    # use_cache: bool = False
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_torch_fused"
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    # predict_with_generate: bool = True
    generation_max_length: int = 256
    prediction_loss_only: bool = True
    
    # Checkpointing
    # save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    
    # Metrics
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 1
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Hub integration
    push_to_hub: bool = True
    hub_strategy: str = "checkpoint"
    hub_model_id: str = HF_MODEL_ID
    
    # Advanced features
    neftune_noise_alpha: Optional[float] = 5.0  # 5.0-15.0 for NEFTune, None to disable
    # weight_decay: float = 0.01
    remove_unused_columns: bool = True
    label_names: List[str] = field(default_factory=lambda: ["labels"])


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 32
    lora_alpha: int = 64
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace Hub integration."""
    
    # Authentication
    use_auth_token: bool = True
    hf_token_env_var: str = "HF_TOKEN"
    pretty_name: str = "Whisper Small - Sinhala ASR Fine-Tuned"
    dataset_args: str = "config: si, split: test"
    model_name: str = HF_MODEL_ID
    tasks: str = "automatic-speech-recognition"


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
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    

# Configuration instance for convenience
CONFIG = Config()