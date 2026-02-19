"""
ASR Model Training Module

This module provides a trainer wrapper for Automatic Speech Recognition (ASR) model
fine-tuning using the Hugging Face Transformers library with optional LoRA 
(Low-Rank Adaptation) for parameter-efficient fine-tuning.

Classes:
    ASRTrainer: Main trainer class for ASR model fine-tuning
"""

from dataclasses import asdict
from typing import Optional, Union, Callable, Any
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig as PEFTLoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

from src.config.config import CONFIG


class ASRTrainer:
    """
    Trainer wrapper for ASR model fine-tuning with optional LoRA support.
    
    This class wraps the Hugging Face Seq2SeqTrainer and provides additional
    functionality for LoRA-based parameter-efficient fine-tuning.
    
    Attributes:
        model (PreTrainedModel): The model being trained (with or without LoRA)
        training_args (Seq2SeqTrainingArguments): Training configuration
        trainer (Seq2SeqTrainer): The underlying Hugging Face trainer
    
    Example:
        >>> asr_trainer = ASRTrainer(
        ...     model=model,
        ...     train_dataset=train_data,
        ...     eval_dataset=eval_data,
        ...     data_collator=collator,
        ...     compute_metrics=metric_fn,
        ...     tokenizer=tokenizer
        ... )
        >>> # Access trainer methods directly
        >>> asr_trainer.trainer.train()
        >>> asr_trainer.trainer.evaluate()
        >>> asr_trainer.trainer.save_model("./output")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: Any,
        compute_metrics: Callable,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize the ASR trainer.
        
        Args:
            model: Pre-trained model to fine-tune
            train_dataset: Training dataset
            eval_dataset: Evaluation/validation dataset
            data_collator: Data collator for batching
            compute_metrics: Function to compute evaluation metrics
            tokenizer: Tokenizer/processor for the model
            
        Note:
            If CONFIG.lora is not None, LoRA will be automatically applied
            to the model for parameter-efficient fine-tuning.
        """
        self.model = model
        
        # Apply LoRA if configured
        if CONFIG.lora is not None:
            self.model = self._apply_lora_to_model()

        # Set up training arguments from config
        self.training_args = Seq2SeqTrainingArguments(**asdict(CONFIG.training))

        # Initialize the trainer
        self.trainer = Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG.runtime.early_stopping_patience)],
        )

    def _apply_lora_to_model(self) -> PreTrainedModel:
        """
        Apply LoRA to the model for parameter-efficient fine-tuning.
        
        This method:
        1. Prepares the model for k-bit training (freezes base parameters)
        2. Applies LoRA adapters based on the configuration
        3. Prints information about trainable parameters
        
        Returns:
            PreTrainedModel: Model with LoRA applied
            
        Raises:
            ValueError: If CONFIG.lora is None or invalid
            
        Note:
            LoRA significantly reduces memory usage and training time by only
            training a small number of additional parameters while keeping
            the base model frozen.
        """
        # Prepare model for training (freezes base model parameters)
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Create PEFT config from the LoRA configuration
        peft_config = PEFTLoraConfig(**asdict(CONFIG.lora))
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters info for verification
        self.model.print_trainable_parameters()
            
        return self.model