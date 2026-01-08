"""
Data Collator component for Whisper ASR.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for sequence-to-sequence speech models.
    Handles input_features and labels independently with appropriate padding.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch.
        
        Args:
            features: List of feature dictionaries containing input_features and labels
            
        Returns:
            Batched dictionary with input_features and labels as tensors
        """
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class DataCollatorComponent:
    """Handles data collator creation for Whisper training."""
    
    def __init__(self):
        """Initialize the data collator component."""
        self.data_collator = None
    
    def create(self, processor: Any, decoder_start_token_id: int) -> DataCollatorSpeechSeq2SeqWithPadding:
        """
        Create a data collator for speech sequence-to-sequence training.
        
        Args:
            processor: WhisperProcessor instance
            decoder_start_token_id: Decoder start token ID from model config
            
        Returns:
            DataCollatorSpeechSeq2SeqWithPadding instance
        """
        logger.info("Creating data collator...")
        
        try:
            self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
                processor=processor,
                decoder_start_token_id=decoder_start_token_id,
            )
            logger.info("Data collator created successfully")
            return self.data_collator
            
        except Exception as e:
            logger.error(f"Failed to create data collator: {str(e)}")
            raise
    
    def get(self) -> DataCollatorSpeechSeq2SeqWithPadding:
        """Get the data collator instance."""
        if self.data_collator is None:
            logger.error("Data collator not created. Call create() first.")
            raise ValueError("Data collator not initialized")
        return self.data_collator
