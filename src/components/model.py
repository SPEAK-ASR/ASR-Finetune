"""
Model component for Whisper ASR.
"""

import torch
from transformers import WhisperForConditionalGeneration
from src.utils.logger import setup_logger
from src.config.config import CONFIG

logger = setup_logger(__name__)


class ModelComponent:
    """Handles Whisper model loading and configuration."""
    
    def __init__(self, model_name: str, language: str, task: str):
        """
        Initialize the model component.
        
        Args:
            model_name: Pretrained Whisper model name
            language: Target language for transcription
            task: Task type (transcribe or translate)
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.model = None
    
    def load(self) -> WhisperForConditionalGeneration:
        """
        Load the pre-trained Whisper model and configure generation settings.
        
        Returns:
            WhisperForConditionalGeneration instance
        """
        logger.info(f"Loading Whisper model from {self.model_name}...")
        
        try:
            # Load model in float32 by default
            # The Trainer will handle dtype conversion for bf16/fp16 training automatically
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=CONFIG.paths.model_cache_dir
            )
            
            # Configure generation settings
            self.model.generation_config.language = self.language.lower()
            self.model.generation_config.task = self.task
            self.model.generation_config.forced_decoder_ids = None
            
            # Set pad_token_id to avoid attention_mask warnings during generation
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.model.config.eos_token_id
            
            logger.info(
                f"Model loaded successfully with language={self.language}, "
                f"task={self.task}"
            )
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get(self) -> WhisperForConditionalGeneration:
        """Get the model instance."""
        if self.model is None:
            return self.load()
        return self.model
