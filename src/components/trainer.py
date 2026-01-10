from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from typing import Optional
from ..config.lora_config import LoRAConfig, apply_lora_to_model

class ASRTrainerConfig:
    """Configuration for ASR training arguments."""

    def __init__(
        self,
        run_name: Optional[str] = None,
        output_dir: str = "./whisper-small-si",
        per_device_train_batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        max_steps: int = -1,
        num_train_epochs: Optional[int] = None,
        gradient_checkpointing: bool = True,
        fp16: bool = False,
        bf16: bool = True,
        auto_find_batch_size: bool = False,
        optim: str = "adamw_torch",
        eval_strategy: str = "steps",
        per_device_eval_batch_size: int = 8,
        predict_with_generate: bool = True,
        generation_max_length: int = 225,
        save_steps: int = 1000,
        eval_steps: int = 1000,
        logging_steps: int = 100,
        report_to: list = None,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "wer",
        greater_is_better: bool = False,
        push_to_hub: bool = True,
        neftune_noise_alpha: Optional[float] = None,
        use_cache: bool = False,
        remove_unused_columns: bool = True,
        **kwargs
    ):
        """
        Initialize training configuration.

        Args:
            run_name: Descriptor for the run (used for logging in wandb, mlflow, tensorboard, etc.)
            output_dir: Local directory to save model weights and Hub repository name
            per_device_train_batch_size: Batch size per device during training
            gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass
            learning_rate: Initial learning rate
            warmup_steps: Number of steps for learning rate warmup
            max_steps: Maximum number of training steps
            num_train_epochs: Total number of training epochs to perform
            gradient_checkpointing: Enable gradient checkpointing to save memory
            fp16: Enable mixed precision training (FP16)
            bf16: Enable mixed precision training (BF16)
            auto_find_batch_size: Automatically find batch size that fits in memory through exponential decay
            optim: Optimizer to use (e.g., "adamw_torch", "adamw_hf", etc.)
            eval_strategy: Evaluation strategy to adopt during training
            per_device_eval_batch_size: Batch size per device during evaluation
            predict_with_generate: Use generation for evaluation
            generation_max_length: Maximum number of tokens to generate during evaluation
            save_steps: Save checkpoint every X steps
            eval_steps: Evaluate every X steps
            logging_steps: Log every X steps
            report_to: List of integrations to report results to (e.g., ["tensorboard", "mlflow"])
            load_best_model_at_end: Load the best model at the end of training
            metric_for_best_model: Metric to use for model selection
            greater_is_better: Whether a larger metric value is better
            push_to_hub: Push model to Hugging Face Hub
            neftune_noise_alpha: Activate NEFTune noise embeddings for improved fine-tuning (5.0-15.0 recommended, None to disable)
            use_cache: Whether the model should use the past key/values attentions (if applicable to the model) to speed up decoding
            **kwargs: Additional arguments to pass to Seq2SeqTrainingArguments
        """
        self.run_name = run_name
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.gradient_checkpointing = gradient_checkpointing
        self.fp16 = fp16
        self.bf16 = bf16
        self.auto_find_batch_size = auto_find_batch_size
        self.optim = optim
        self.eval_strategy = eval_strategy
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.predict_with_generate = predict_with_generate
        self.generation_max_length = generation_max_length
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.report_to = report_to if report_to is not None else ["tensorboard"]
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.push_to_hub = push_to_hub
        self.neftune_noise_alpha = neftune_noise_alpha
        self.use_cache = use_cache
        self.remove_unused_columns = remove_unused_columns
        self.kwargs = kwargs

    def to_training_arguments(self) -> Seq2SeqTrainingArguments:
        """
        Convert configuration to Seq2SeqTrainingArguments.

        Returns:
            Seq2SeqTrainingArguments object
        """
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            num_train_epochs=self.num_train_epochs,
            gradient_checkpointing=self.gradient_checkpointing,
            fp16=self.fp16,
            bf16=self.bf16,
            auto_find_batch_size=self.auto_find_batch_size,
            optim=self.optim,
            eval_strategy=self.eval_strategy,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            predict_with_generate=self.predict_with_generate,
            generation_max_length=self.generation_max_length,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            logging_steps=self.logging_steps,
            report_to=self.report_to,
            run_name=self.run_name,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            push_to_hub=self.push_to_hub,
            neftune_noise_alpha=self.neftune_noise_alpha,
            use_cache=self.use_cache,
            remove_unused_columns=self.remove_unused_columns,
            **self.kwargs
        )


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

    def train(self):
        """Start the training process."""
        return self.trainer.train()

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
    config: Optional[ASRTrainerConfig] = None,
    training_args: Optional[Seq2SeqTrainingArguments] = None,
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
        config: Training configuration. If None, uses default configuration
        training_args: Direct training arguments. If provided, overrides config
        lora_config: LoRA configuration. If provided, applies LoRA to the model

    Returns:
        ASRTrainer instance
    """
    # Apply LoRA if configuration is provided
    # if lora_config is not None:
    #     model = apply_lora_to_model(model, lora_config)

    if training_args is None:
        if config is None:
            config = ASRTrainerConfig()
        training_args = config.to_training_arguments()

    return ASRTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
