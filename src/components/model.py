"""
Model component for Whisper ASR.
"""

from transformers import WhisperForConditionalGeneration
from src.utils.logger import setup_logger

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
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name
            )
            
            # Configure generation settings
            self.model.generation_config.language = self.language.lower()
            self.model.generation_config.task = self.task
            self.model.generation_config.forced_decoder_ids = None
            
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
