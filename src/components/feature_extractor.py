"""
Feature Extractor component for Whisper ASR.
"""

from transformers import WhisperFeatureExtractor
from src.utils.logger import setup_logger
from src.config.config import CONFIG

logger = setup_logger(__name__)


class FeatureExtractorComponent:
    """Handles Whisper feature extraction."""
    
    def __init__(self, model_name: str):
        """
        Initialize the feature extractor component.
        
        Args:
            model_name: Pretrained Whisper model name
        """
        self.model_name = model_name
        self.feature_extractor = None
    
    def load(self) -> WhisperFeatureExtractor:
        """
        Load the Whisper feature extractor.
        
        Returns:
            WhisperFeatureExtractor instance
        """
        logger.info(f"Loading feature extractor from {self.model_name}...")
        
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_name,
                cache_dir=CONFIG.paths.model_cache_dir
            )
            logger.info("Feature extractor loaded successfully")
            return self.feature_extractor
            
        except Exception as e:
            logger.error(f"Failed to load feature extractor: {str(e)}")
            raise
    
    def get(self) -> WhisperFeatureExtractor:
        """Get the feature extractor instance."""
        if self.feature_extractor is None:
            return self.load()
        return self.feature_extractor
