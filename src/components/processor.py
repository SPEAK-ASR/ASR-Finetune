"""
Processor component for Whisper ASR.
"""

from transformers import WhisperProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProcessorComponent:
    """Handles Whisper processor creation."""
    
    def __init__(self, model_name: str, language: str, task: str):
        """
        Initialize the processor component.
        
        Args:
            model_name: Pretrained Whisper model name
            language: Target language for transcription
            task: Task type (transcribe or translate)
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.processor = None
    
    def create(self) -> WhisperProcessor:
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
    
    def get(self) -> WhisperProcessor:
        """Get the processor instance."""
        if self.processor is None:
            return self.create()
        return self.processor
