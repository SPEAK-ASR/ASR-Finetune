"""
Tokenizer component for Whisper ASR.
"""

from transformers import WhisperTokenizer
from datasets import DatasetDict
from src.utils.logger import setup_logger
from src.config.config import CONFIG

logger = setup_logger(__name__)


class TokenizerComponent:
    """Handles Whisper tokenization."""
    
    def __init__(self, model_name: str, language: str, task: str):
        """
        Initialize the tokenizer component.
        
        Args:
            model_name: Pretrained Whisper model name
            language: Target language for transcription
            task: Task type (transcribe or translate)
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.tokenizer = None
    
    def load(self) -> WhisperTokenizer:
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
                task=self.task,
                cache_dir=CONFIG.paths.model_cache_dir
            )
            logger.info("Tokenizer loaded successfully")
            return self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def get(self) -> WhisperTokenizer:
        """Get the tokenizer instance."""
        if self.tokenizer is None:
            return self.load()
        return self.tokenizer
    
    def verify(self, dataset: DatasetDict, split: str = "train", index: int = 0) -> bool:
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
            logger.error("Tokenizer not loaded. Call load() first.")
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
