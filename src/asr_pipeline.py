"""
ASR Pipeline Facade for Whisper fine-tuning.
Provides a simplified interface for the complete fine-tuning workflow.
"""

from typing import Optional
from datasets import DatasetDict, Dataset

from src.components.feature_extractor import FeatureExtractorComponent
from src.components.tokenizer import TokenizerComponent
from src.components.processor import ProcessorComponent
from src.components.model import ModelComponent
from src.components.data_collator import DataCollatorComponent
from src.components.trainer import create_trainer, ASRTrainerConfig
from src.components.evaluator import ASREvaluator
from src.config.lora_config import LoRAConfig
from src.data_preprocessor import DataPreprocessor
from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WhisperASRPipeline:
    """
    Facade for Whisper ASR fine-tuning pipeline.
    Simplifies the complex workflow into easy-to-use high-level methods.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "Sinhala",
        task: str = "transcribe"
    ):
        """
        Initialize the ASR pipeline facade.
        
        Args:
            model_name: Pretrained Whisper model name
            language: Target language for transcription
            task: Task type (transcribe or translate)
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self._initialized = False
        
        # Initialize component wrappers (internal complexity hidden)
        self._feature_extractor = FeatureExtractorComponent(model_name)
        self._tokenizer = TokenizerComponent(model_name, language, task)
        self._processor = ProcessorComponent(model_name, language, task)
        self._model = ModelComponent(model_name, language, task)
        self._data_collator = DataCollatorComponent()
        self._preprocessor = DataPreprocessor()
        
        logger.info(f"Pipeline created - Model: {model_name}, Language: {language}")
    
    def initialize(self, dataset: Optional[DatasetDict] = None):
        """
        Initialize all pipeline components. Call this before training.
        
        Args:
            dataset: Optional dataset for tokenizer verification
            
        Returns:
            Self for method chaining
        """
        if self._initialized:
            logger.warning("Pipeline already initialized")
            return self
        
        logger.info("Initializing pipeline components...")
        
        self._feature_extractor.load()
        self._tokenizer.load()
        
        if dataset:
            self._tokenizer.verify(dataset)
        
        self._processor.create()
        
        self._initialized = True
        logger.info("Pipeline initialization complete")
        return self
    
    def prepare_data(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare dataset for training (feature extraction + tokenization).
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Prepared dataset ready for training
        """
        self._ensure_initialized()
        logger.info("Preparing dataset...")
        
        prepared = self._preprocessor.prepare_dataset(
            dataset=dataset,
            feature_extractor_component=self._feature_extractor,
            tokenizer_component=self._tokenizer
        )
        
        logger.info("Dataset preparation complete")
        return prepared
    
    def finetune(self, dataset: DatasetDict) -> dict:
        """
        High-level method to finetune the model with sensible defaults.

        Args:
            dataset: Prepared dataset for training
            
        Returns:
            Training results
        """
        self._ensure_initialized()
        logger.info("Starting fine-tuning workflow...")
        
        # Configure training
        training_config = ASRTrainerConfig(
            run_name=CONFIG.training.run_name,
            output_dir=CONFIG.training.output_dir,
            num_train_epochs=CONFIG.training.num_train_epochs,
            max_steps=CONFIG.training.max_steps,
            per_device_train_batch_size=CONFIG.training.per_device_train_batch_size,
            per_device_eval_batch_size=CONFIG.training.per_device_eval_batch_size,
            gradient_accumulation_steps=CONFIG.training.gradient_accumulation_steps,
            auto_find_batch_size=CONFIG.training.auto_find_batch_size,
            learning_rate=CONFIG.training.learning_rate,
            warmup_steps=CONFIG.training.warmup_steps,
            lr_scheduler_type=CONFIG.training.lr_scheduler_type,
            gradient_checkpointing=CONFIG.training.gradient_checkpointing,
            fp16=CONFIG.training.fp16,
            bf16=CONFIG.training.bf16,
            optim=CONFIG.training.optim,
            eval_strategy=CONFIG.training.eval_strategy,
            eval_steps=CONFIG.training.eval_steps,
            predict_with_generate=CONFIG.training.predict_with_generate,
            save_strategy=CONFIG.training.save_strategy,
            save_steps=CONFIG.training.save_steps,
            load_best_model_at_end=CONFIG.training.load_best_model_at_end,
            metric_for_best_model=CONFIG.training.metric_for_best_model,
            greater_is_better=CONFIG.training.greater_is_better,
            logging_strategy=CONFIG.training.logging_strategy,
            logging_steps=CONFIG.training.logging_steps,
            logging_first_step=CONFIG.training.logging_first_step,
            report_to=CONFIG.training.report_to,
            push_to_hub=CONFIG.training.push_to_hub,
            hub_strategy=CONFIG.training.hub_strategy,
            neftune_noise_alpha=CONFIG.training.neftune_noise_alpha,
            weight_decay=CONFIG.training.weight_decay,
            use_cache=CONFIG.training.use_cache,
            remove_unused_columns=CONFIG.training.remove_unused_columns,
            label_names=CONFIG.training.label_names,
        )
        
        # Configure LoRA if enabled
        lora_config = None
        lora_config = LoRAConfig(
            r=CONFIG.lora.r,
            lora_alpha=CONFIG.lora.lora_alpha,
            target_modules=CONFIG.lora.target_modules,
            lora_dropout=CONFIG.lora.lora_dropout,
            bias=CONFIG.lora.bias,
            task_type=CONFIG.lora.task_type,
        )
        
        # Setup data collator
        self._setup_data_collator()
        
        # Create and run trainer
        trainer = self._create_trainer(
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            training_config=training_config,
            lora_config=lora_config,
        )
        
        logger.info("Beginning training...")
        results = trainer.train()
        
        logger.info("Fine-tuning complete!")

        kwargs = {
            "repo_id": CONFIG.huggingface.hub_repo_id,
            "dataset_tags": CONFIG.dataset.dataset_name,
            "dataset": CONFIG.huggingface.pretty_name,  # a 'pretty' name for the training dataset
            "dataset_args": CONFIG.huggingface.dataset_args,
            "language": "si",
            "model_name": CONFIG.huggingface.model_name,  # a 'pretty' name for your model
            "finetuned_from": CONFIG.model.model_name,
            "tasks": CONFIG.huggingface.tasks,
        }
        trainer.push_to_hub(**kwargs)
        logger.info("Model pushed to HuggingFace Hub")

        return results
    
    def _setup_data_collator(self):
        """Internal method to setup data collator."""
        processor = self._processor.get()
        model = self._model.get()
        self._data_collator.create(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    
    def _create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_config: ASRTrainerConfig,
        lora_config: Optional[LoRAConfig] = None,
    ):
        """Internal method to create trainer."""
        model = self._model.get()
        processor = self._processor.get()
        data_collator = self._data_collator.get()
        tokenizer = self._tokenizer.get()
        
        evaluator = ASREvaluator(tokenizer)
        
        return create_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=evaluator.compute_metrics,
            tokenizer=tokenizer,
            config=training_config,
            lora_config=lora_config,
        )
    
    def _ensure_initialized(self):
        """Ensure pipeline is initialized before operations."""
        if not self._initialized:
            logger.warning("Pipeline not initialized. Calling initialize()...")
            self.initialize()
    
    # Legacy methods for backward compatibility and advanced usage
    
    def create_data_collator(self):
        """Create data collator (legacy method)."""
        self._setup_data_collator()
        return self._data_collator.get()
    
    def prepare_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset (legacy method)."""
        return self.prepare_data(dataset)
    
    def get_processor(self):
        """Get processor component."""
        return self._processor.get()
    
    def get_tokenizer(self):
        """Get tokenizer component."""
        return self._tokenizer.get()
    
    def get_feature_extractor(self):
        """Get feature extractor component."""
        return self._feature_extractor.get()
    
    def get_model(self):
        """Get model component."""
        return self._model.get()
    
    def get_data_collator(self):
        """Get data collator component."""
        return self._data_collator.get()
