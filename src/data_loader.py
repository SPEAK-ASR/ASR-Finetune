"""
Data loader for Whisper fine-tuning.
Handles loading and preparation of audio datasets from HuggingFace.
Supports combining multiple datasets into a single DatasetDict.
"""

import os
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, disable_caching
from typing import Optional, List

from src.utils.logger import setup_logger
from src.config.config import CONFIG
from src.config.dataset import SingleDatasetConfig

logger = setup_logger(__name__)


class WhisperDataLoader:
    """
    Manages loading and preparation of audio datasets for Whisper fine-tuning.
    Supports loading and combining multiple datasets.
    """
    
    def __init__(self):
        """
        Initialize the data loader.
        
        Args:
            datasets_config: List of dataset configurations. If None, uses CONFIG.dataset.datasets
            token: Whether to use authentication token for private datasets
        """
        self.datasets_config: List[SingleDatasetConfig] = CONFIG.dataset.datasets
        self.token: bool = CONFIG.dataset.use_auth_token
        self.dataset: Optional[DatasetDict] = None
        
        dataset_names = [ds.dataset_name for ds in self.datasets_config]
        logger.info(f"DataLoader initialized for {len(self.datasets_config)} dataset(s): {dataset_names}")
    
    def load_datasets(self) -> DatasetDict:
        """
        Load and combine training and test datasets from all configured sources.
        
        Returns:
            DatasetDict containing combined train and test splits
        """
        try:
            all_train_datasets = []
            all_test_datasets = []
            
            for ds_config in self.datasets_config:
                logger.info(f"Loading dataset '{ds_config.dataset_name}'")
                
                # Load training data
                if ds_config.train_split is not None:
                    logger.info(f"  Loading train split: {ds_config.train_split}")
                    train_data = load_dataset(
                        ds_config.dataset_name,
                        split=ds_config.train_split,
                        token=self.token,
                        cache_dir=CONFIG.paths.cache_dir,
                        keep_in_memory=CONFIG.dataset.keep_in_memory
                    )
                    all_train_datasets.append(train_data)
                    logger.info(f"  Train data loaded: {len(train_data)} samples")
                
                # Load test data
                if ds_config.test_split is not None:
                    logger.info(f"  Loading test split: {ds_config.test_split}")
                    test_data = load_dataset(
                        ds_config.dataset_name,
                        split=ds_config.test_split,
                        token=self.token,
                        cache_dir=CONFIG.paths.cache_dir,
                        keep_in_memory=CONFIG.dataset.keep_in_memory
                    )
                    all_test_datasets.append(test_data)
                    logger.info(f"  Test data loaded: {len(test_data)} samples")
            
            # Combine datasets
            self.dataset = DatasetDict()
            
            if len(all_train_datasets) == 1:
                self.dataset["train"] = all_train_datasets[0]
                self.dataset["test"] = all_test_datasets[0]
            else:
                logger.info(f"Combining {len(all_train_datasets)} datasets...")
                self.dataset["train"] = concatenate_datasets(all_train_datasets)
                self.dataset["test"] = concatenate_datasets(all_test_datasets)
            
            logger.info(f"Combined train dataset: {len(self.dataset['train'])} samples")
            logger.info(f"Combined test dataset: {len(self.dataset['test'])} samples")
            
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

    def load_parallel_text_dataset(
        self,
        dataset_name: Optional[str] = None,
    ) -> Optional[DatasetDict]:
        """Load the parallel noisy-text / clean-text corpus used in Stage 1.

        The dataset is expected to expose ``noisy_text`` and ``clean_text`` columns
        (configurable via ``CONFIG.dataset.parallel_noisy_column`` and
        ``CONFIG.dataset.parallel_clean_column``).

        Returns None if no parallel dataset is configured.
        """
        name = dataset_name or CONFIG.dataset.parallel_text_dataset
        if name is None:
            logger.info("No parallel text dataset configured - skipping")
            return None

        logger.info(f"Loading parallel text dataset '{name}'")
        try:
            ds = load_dataset(
                name,
                token=self.token,
                cache_dir=CONFIG.paths.cache_dir,
                keep_in_memory=CONFIG.dataset.keep_in_memory,
            )
        except Exception as e:
            logger.error(f"Failed to load parallel text dataset '{name}': {e}")
            raise

        if isinstance(ds, Dataset):
            ds = DatasetDict({"train": ds})

        logger.info(f"Parallel text dataset loaded: { {k: len(v) for k, v in ds.items()} }")
        return ds

    def load_pseudo_dataset(
        self,
        dataset_name: Optional[str] = None,
    ) -> DatasetDict:
        """Load the Stage-0 pseudo dataset (audio + asr_hyp_text + clean_text).

        Used both in Stage 1 (text-only) and Stage 2 (joint training).
        """
        name = dataset_name or CONFIG.dataset.pseudo_dataset_name
        logger.info(f"Loading pseudo dataset '{name}'")
        try:
            ds = load_dataset(
                name,
                token=self.token,
                cache_dir=CONFIG.paths.cache_dir,
                keep_in_memory=CONFIG.dataset.keep_in_memory,
            )
        except Exception as e:
            logger.error(f"Failed to load pseudo dataset '{name}': {e}")
            raise

        if isinstance(ds, Dataset):
            ds = DatasetDict({"train": ds})

        logger.info(f"Pseudo dataset loaded: { {k: len(v) for k, v in ds.items()} }")
        return ds
