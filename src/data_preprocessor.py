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

    def prepare_joint_dataset(
        self,
        dataset: DatasetDict,
        feature_extractor_component,
        whisper_tokenizer_component,
        byt5_tokenizer,
        hyp_column: Optional[str] = None,
        clean_column: Optional[str] = None,
    ) -> DatasetDict:
        """Prepare a dataset for the joint ASR+PostProc pipeline.

        The input dataset (from Stage 0) must contain:
          - ``audio`` column (16kHz waveform)
          - hyp_column (default ``asr_hyp_text``): the ASR's hypothesised transcript
          - clean_column (default ``clean_text``): the gold / cleaned transcript

        Produces columns:
          - ``input_features``: log-Mel features for Whisper
          - ``asr_hyp_labels``: Whisper tokenizer ids for the hypothesis text
            (teacher-forced into Whisper's decoder so hidden states encode Whisper's
            actual error pattern)
          - ``clean_labels``: ByT5 byte ids for the clean target text (main loss)
        """
        hyp_col = hyp_column or CONFIG.dataset.pseudo_hyp_column
        clean_col = clean_column or CONFIG.dataset.pseudo_clean_column

        feature_extractor = feature_extractor_component.get()
        whisper_tokenizer = whisper_tokenizer_component.get()

        def _map(batch):
            audio = batch["audio"]
            batch["input_features"] = feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).input_features[0]

            # Whisper-side label = ASR hypothesis (for teacher-forcing and aux loss)
            batch["asr_hyp_labels"] = whisper_tokenizer(
                batch[hyp_col]
            ).input_ids

            # Post-processor-side label = clean transcript (main loss target)
            batch["clean_labels"] = byt5_tokenizer(
                batch[clean_col],
                truncation=True,
                max_length=CONFIG.postprocessor.max_target_length,
            ).input_ids

            return batch

        logger.info("Preparing joint dataset (features + dual labels)...")
        keep_cols = {"input_features", "asr_hyp_labels", "clean_labels"}
        all_cols = set(dataset["train"].column_names)
        remove_cols = list(all_cols - keep_cols) if all_cols else None

        prepared = dataset.map(_map, remove_columns=remove_cols)
        logger.info("Joint dataset preparation complete")
        return prepared

    def prepare_parallel_text_dataset(
        self,
        dataset: DatasetDict,
        byt5_tokenizer,
        noisy_column: Optional[str] = None,
        clean_column: Optional[str] = None,
    ) -> DatasetDict:
        """Prepare a plain text-to-text parallel dataset for Stage 1 post-proc pretraining.

        Produces ``input_ids`` and ``labels`` columns suitable for a seq2seq
        ``DataCollatorForSeq2Seq`` with the ByT5 tokenizer.
        """
        noisy_col = noisy_column or CONFIG.dataset.parallel_noisy_column
        clean_col = clean_column or CONFIG.dataset.parallel_clean_column

        def _map(batch):
            src = byt5_tokenizer(
                batch[noisy_col],
                truncation=True,
                max_length=CONFIG.postprocessor.max_target_length,
            )
            tgt = byt5_tokenizer(
                batch[clean_col],
                truncation=True,
                max_length=CONFIG.postprocessor.max_target_length,
            )
            return {
                "input_ids": src["input_ids"],
                "attention_mask": src["attention_mask"],
                "labels": tgt["input_ids"],
            }

        all_cols = set(dataset["train"].column_names)
        keep = {"input_ids", "attention_mask", "labels"}
        remove_cols = list(all_cols - keep)

        logger.info("Preparing parallel text dataset for post-processor pretraining...")
        return dataset.map(_map, remove_columns=remove_cols, batched=False)
