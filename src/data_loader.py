"""
Data loader for Whisper fine-tuning.
Handles loading and preparation of audio datasets from HuggingFace.
"""

from datasets import load_dataset, DatasetDict
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WhisperDataLoader:
    """
    Manages loading and preparation of audio datasets for Whisper fine-tuning.
    """
    
    def __init__(
        self,
        dataset_name: str,
        token: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            token: Whether to use authentication token for private datasets
        """
        self.dataset_name = dataset_name
        self.token = token
        self.dataset: Optional[DatasetDict] = None
        
        logger.info(f"DataLoader initialized for dataset: {dataset_name}")
    
    def load_datasets(
        self,
        train_split: str = "train+validation",
        test_split: str = "test"
    ) -> DatasetDict:
        """
        Load training and test datasets.
        
        Args:
            train_split: Split specification for training data
            test_split: Split specification for test data
            
        Returns:
            DatasetDict containing train and test splits
        """
        try:
            logger.info(f"Loading dataset '{self.dataset_name}'")
            
            self.dataset = DatasetDict()
            
            # Load training data
            logger.info(f"Loading train split: {train_split}")
            self.dataset["train"] = load_dataset(
                self.dataset_name,
                split=train_split,
                token=self.token
            )
            logger.info(f"Train dataset loaded: {len(self.dataset['train'])} samples")
            
            # Load test data
            logger.info(f"Loading test split: {test_split}")
            self.dataset["test"] = load_dataset(
                self.dataset_name,
                split=test_split,
                token=self.token
            )
            logger.info(f"Test dataset loaded: {len(self.dataset['test'])} samples")
            
            self._log_dataset_info()
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
        
    
    def _log_dataset_info(self) -> None:
        """Log detailed information about the loaded dataset."""
        if self.dataset is None:
            logger.warning("No dataset loaded yet")
            return
        
        logger.info("=" * 50)
        logger.info("Dataset Summary:")
        for split_name, split_data in self.dataset.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")
            logger.info(f"  Features: {list(split_data.features.keys())}")
        logger.info("=" * 50)


    def get_dataset(self) -> Optional[DatasetDict]:
        """
        Get the loaded dataset.
        
        Returns:
            DatasetDict if loaded, None otherwise
        """
        return self.dataset
