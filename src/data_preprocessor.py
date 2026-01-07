"""
Data preprocessing for Whisper fine-tuning.
Handles dataset transformations and column operations.
"""

from datasets import DatasetDict, Dataset
from typing import List, Optional, Union

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
    
    def get_dataset(self) -> Optional[DatasetDict]:
        """
        Get the preprocessed dataset.
        
        Returns:
            DatasetDict or None if not set
        """
        return self.dataset
