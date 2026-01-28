from dataclasses import asdict
from src.config.config import CONFIG
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from typing import Optional, Union
from ..config.lora_config import LoRAConfig, apply_lora_to_model


class ASRTrainerConfig:
    """Configuration for ASR training arguments."""

    def get_training_arguments(self) -> Seq2SeqTrainingArguments:
        """
        Create Seq2SeqTrainingArguments from configuration.
        Automatically passes all TrainingConfig fields to Seq2SeqTrainingArguments.

        Returns:
            Seq2SeqTrainingArguments object
        """
        # Convert TrainingConfig dataclass to dict and pass to Seq2SeqTrainingArguments
        training_config_dict = asdict(CONFIG.training)
        return Seq2SeqTrainingArguments(**training_config_dict)


class ASRTrainer:
    """Trainer wrapper for ASR model fine-tuning."""

    def __init__(
        self,
        model,
        training_args: Seq2SeqTrainingArguments,
        train_dataset,
        eval_dataset,
        data_collator,
        compute_metrics,
        tokenizer,
    ):
        """
        Initialize the ASR trainer.

        Args:
            model: The Whisper model to train
            training_args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            compute_metrics: Function to compute evaluation metrics
            tokenizer: Feature extractor/tokenizer for processing
        """
        self.trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        model.config.use_cache = False

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        """Start the training process."""
        # Use provided value or fall back to config
        checkpoint = resume_from_checkpoint if resume_from_checkpoint is not None else CONFIG.runtime.resume_from_checkpoint
        return self.trainer.train(resume_from_checkpoint=checkpoint)

    def evaluate(self):
        """Evaluate the model."""
        return self.trainer.evaluate()

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the trained model.

        Args:
            output_dir: Directory to save the model. If None, uses the default output directory.
        """
        self.trainer.save_model(output_dir)

    def push_to_hub(self, commit_message: Optional[str] = None, **kwargs):
        """
        Push the model to Hugging Face Hub.

        Args:
            commit_message: Commit message for the push
            **kwargs: Additional model card kwargs (dataset_tags, dataset, language, etc.)
        """
        self.trainer.push_to_hub(commit_message=commit_message, **kwargs)


def create_trainer(
    model,
    train_dataset,
    eval_dataset,
    data_collator,
    compute_metrics,
    tokenizer,
    lora_config: Optional[LoRAConfig] = None,
) -> ASRTrainer:
    """
    Factory function to create an ASR trainer.

    Args:
        model: The Whisper model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching
        compute_metrics: Function to compute evaluation metrics
        tokenizer: Tokenizer for processing
        lora_config: LoRA configuration. If provided, applies LoRA to the model

    Returns:
        ASRTrainer instance
    """
    # Apply LoRA if configuration is provided
    if lora_config is not None:
        model = apply_lora_to_model(model, lora_config)

    training_args = ASRTrainerConfig().get_training_arguments()

    return ASRTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
