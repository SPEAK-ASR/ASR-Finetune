"""
ASR Pipeline components for Whisper fine-tuning.
Handles feature extraction, tokenization, and processor creation.
"""

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor
)
from typing import Optional
from datasets import DatasetDict, Dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WhisperASRPipeline:
    """
    Manages the ASR pipeline components for Whisper fine-tuning.
    Handles feature extractor, tokenizer, and processor initialization.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "Sinhala",
        task: str = "transcribe"
    ):
        """
        Initialize the ASR pipeline.
        
        Args:
            model_name: Pretrained Whisper model name
            language: Target language for transcription
            task: Task type (transcribe or translate)
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        
        self.feature_extractor: Optional[WhisperFeatureExtractor] = None
        self.tokenizer: Optional[WhisperTokenizer] = None
        self.processor: Optional[WhisperProcessor] = None
        
        logger.info(
            f"ASR Pipeline initialized - Model: {model_name}, "
            f"Language: {language}, Task: {task}"
        )
    
    def _load_feature_extractor(self) -> WhisperFeatureExtractor:
        """
        Load the Whisper feature extractor.
        
        Returns:
            WhisperFeatureExtractor instance
        """
        logger.info(f"Loading feature extractor from {self.model_name}...")
        
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_name
            )
            logger.info("Feature extractor loaded successfully")
            return self.feature_extractor
            
        except Exception as e:
            logger.error(f"Failed to load feature extractor: {str(e)}")
            raise
    
    def _load_tokenizer(self) -> WhisperTokenizer:
        """
        Load the Whisper tokenizer with language and task configuration.
        
        Returns:
            WhisperTokenizer instance
        """
        logger.info(
            f"Loading tokenizer from {self.model_name} "
            f"(Language: {self.language}, Task: {self.task})..."
        )
        
        try:
            self.tokenizer = WhisperTokenizer.from_pretrained(
                self.model_name,
                language=self.language,
                task=self.task
            )
            logger.info("Tokenizer loaded successfully")
            return self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def verify_tokenizer(self, dataset: DatasetDict, split: str = "train", index: int = 0) -> bool:
        """
        Verify tokenizer by encoding and decoding a sample from the dataset.
        
        Args:
            dataset: Dataset containing text samples
            split: Dataset split to use for verification
            index: Index of sample to verify
            
        Returns:
            True if tokenizer correctly encodes/decodes, False otherwise
        """
        if self.tokenizer is None:
            logger.error("Tokenizer not loaded. Call load_tokenizer() first.")
            return False
        
        try:
            # Get sample text
            input_str = dataset[split][index]["text"]
            logger.info(f"Verifying tokenizer with sample from {split}[{index}]")
            
            # Encode and decode
            labels = self.tokenizer(input_str).input_ids
            decoded_with_special = self.tokenizer.decode(labels, skip_special_tokens=False)
            decoded_str = self.tokenizer.decode(labels, skip_special_tokens=True)
            
            # Log results
            logger.info(f"Input:                 {input_str}")
            logger.info(f"Decoded w/ special:    {decoded_with_special}")
            logger.info(f"Decoded w/out special: {decoded_str}")
            
            are_equal = input_str == decoded_str
            logger.info(f"Are equal:             {are_equal}")
            
            if are_equal:
                logger.info("✓ Tokenizer verification successful")
            else:
                logger.warning("✗ Tokenizer verification failed - decoded text differs from input")
            
            return are_equal
            
        except Exception as e:
            logger.error(f"Tokenizer verification failed: {str(e)}")
            return False
    
    def _create_processor(self) -> WhisperProcessor:
        """
        Create a WhisperProcessor combining feature extractor and tokenizer.
        
        Returns:
            WhisperProcessor instance
        """
        logger.info(
            f"Creating Whisper processor from {self.model_name} "
            f"(Language: {self.language}, Task: {self.task})..."
        )
        
        try:
            self.processor = WhisperProcessor.from_pretrained(
                self.model_name,
                language=self.language,
                task=self.task
            )
            logger.info("Processor created successfully")
            return self.processor
            
        except Exception as e:
            logger.error(f"Failed to create processor: {str(e)}")
            raise
    
    def setup_pipeline(self, dataset: Optional[DatasetDict] = None) -> WhisperProcessor:
        """
        Setup complete pipeline: load components and optionally verify tokenizer.
        
        Args:
            dataset: Optional dataset for tokenizer verification
            
        Returns:
            WhisperProcessor instance
        """
        logger.info("Setting up complete ASR pipeline...")
        
        # Load feature extractor
        self._load_feature_extractor()
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Verify tokenizer if dataset provided
        if dataset:
            self.verify_tokenizer(dataset)
        
        # Create processor
        processor = self._create_processor()
        
        logger.info("ASR pipeline setup complete")
        return processor
    
    def prepare_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Prepare dataset by applying feature extraction and tokenization.
        
        Args:
            dataset: DatasetDict to prepare
        Returns:
            Prepared DatasetDict
        """
        dataset = dataset.map(self._prepare_data, remove_columns=dataset.column_names["train"])
        return dataset
    
    def _prepare_data(self, batch: Dataset) -> Dataset:
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    
    def get_processor(self) -> Optional[WhisperProcessor]:
        """Get the processor if created."""
        return self.processor
    
    def get_tokenizer(self) -> Optional[WhisperTokenizer]:
        """Get the tokenizer if loaded."""
        return self.tokenizer
    
    def get_feature_extractor(self) -> Optional[WhisperFeatureExtractor]:
        """Get the feature extractor if loaded."""
        return self.feature_extractor
