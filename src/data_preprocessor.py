"""
Data preprocessing for Whisper fine-tuning.
Handles dataset transformations and column operations.
"""

from datasets import DatasetDict, Dataset, Audio
from typing import List, Optional, Union
from pympler import asizeof

from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """
    Handles preprocessing operations on datasets for Whisper fine-tuning.
    """
    
    def __init__(self, dataset: Optional[DatasetDict] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            dataset: DatasetDict to preprocess. Can be set later via set_dataset()
        """
        self.dataset = dataset
        logger.info("DataPreprocessor initialized")
    
    def set_dataset(self, dataset: DatasetDict) -> None:
        """
        Set the dataset to preprocess.
        
        Args:
            dataset: DatasetDict to preprocess
        """
        self.dataset = dataset
        logger.info("Dataset set for preprocessing")
    
    def remove_columns(
        self,
        columns_to_remove: List[str],
    ) -> DatasetDict:
        """
        Remove specified columns from the dataset.
        
        Args:
            columns_to_remove: List of column names to remove
            
        Returns:
            DatasetDict with specified columns removed
            
        Raises:
            ValueError: If dataset is not set
        """
        if self.dataset is None:
            logger.error("No dataset set. Call set_dataset() first.")
            raise ValueError("No dataset set for preprocessing")
        
        logger.info(f"Removing columns: {columns_to_remove}")
        
        try:
            self.dataset = self.dataset.remove_columns(columns_to_remove)
        except Exception as e:
            logger.error(f"Error removing columns: {str(e)}")
            raise e
        
        logger.info("Column removal complete")
        return self.dataset
    
    def set_sample_rate(self, audio_field_label: str, sample_rate: int) -> None:
        if self.dataset is None:
            logger.warning("Dataset not loaded. Call load_datasets() first.")
            return
        
        try:
            logger.info(f"Setting sample rate to {sample_rate}")
            self.dataset = self.dataset.cast_column(
                audio_field_label,
                Audio(sampling_rate=sample_rate)
            )
            logger.info("Sample rate set successfully.")
        except Exception as e:
            logger.error(f"Failed to set sample rate: {str(e)}")
            raise
    
    def get_dataset(self) -> Optional[DatasetDict]:
        """
        Get the preprocessed dataset.
        
        Returns:
            DatasetDict or None if not set
        """
        return self.dataset
    
    def prepare_dataset(
        self,
        dataset: DatasetDict,
        feature_extractor_component,
        tokenizer_component,
    ) -> DatasetDict:
        """
        Prepare dataset by applying feature extraction and tokenization.
        
        Args:
            dataset: DatasetDict to prepare
            feature_extractor_component: Feature extractor component
            tokenizer_component: Tokenizer component
            
        Returns:
            Prepared DatasetDict
        """
        self.tokenizer = tokenizer_component.get()

        dataset = dataset.filter(
            self._check_token_length,
            input_columns=["text"],
            load_from_cache_file=True,
            desc="Filtering long samples",
        )

        logger.info(f"Dataset size (before): ({self._measure_size(dataset):.2f} MB)")

        prepared_dataset = dataset.map(
            lambda batch: self._prepare_data(
                batch,
                feature_extractor_component,
                tokenizer_component
            ),
            remove_columns=dataset.column_names["train"],
        )

        logger.info(f"Dataset size (after): ({self._measure_size(prepared_dataset):.2f} MB)")

        return prepared_dataset
    
    def _measure_size(self, obj) -> float:
        """Measure size of object in MB."""
        total_size = asizeof.asizeof(obj)
        size_mb = total_size / (1024 * 1024)
        return size_mb
    
    def _check_token_length(self, text: str) -> bool:
        """
        Check if the text will produce a valid token length.
        
        Args:
            text: Dictionary containing 'text' key
            
        Returns:
            True if token length is acceptable, False otherwise
        """
        # Tokenize to check length
        token_ids = self.tokenizer(text).input_ids
        token_length = len(token_ids)
        
        # Return True to keep the sample, False to filter it out
        if token_length > CONFIG.model.max_token_length:
            logger.warning(
                f"⚠️  Filtering out sample with {token_length} tokens (max: {CONFIG.model.max_token_length})")
            return False
        
        return True
    
    def _prepare_data(
        self,
        batch: Dataset,
        feature_extractor_component,
        tokenizer_component
    ) -> Dataset:
        """
        Prepare individual batch: extract features and tokenize text.
        
        Args:
            batch: Batch to prepare
            feature_extractor_component: Feature extractor component
            tokenizer_component: Tokenizer component
            
        Returns:
            Prepared batch
        """
        # Load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # Compute log-Mel input features from input audio array
        feature_extractor = feature_extractor_component.get()
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode target text to label ids
        tokenizer = tokenizer_component.get()
        batch["labels"] = tokenizer(batch["text"]).input_ids

        if len(batch["labels"]) > CONFIG.model.max_token_length:
            logger.warning("tokenized label length exceeds max_token_length")
            logger.warning("need to remove this sample from dataset")
        
        return batch
