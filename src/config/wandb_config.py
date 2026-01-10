"""
Weights & Biases integration for experiment tracking.
Handles authentication with W&B.
"""

import wandb
from typing import Optional
import os

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WandbAuthenticator:
    """Handles authentication with Weights & Biases."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._authenticated = False
        logger.info("WandbAuthenticator initialized")
    
    def authenticate(self) -> bool:
        """
        Authenticate with W&B.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            logger.info("Attempting to authenticate with W&B...")
            
            if self.api_key:
                wandb.login(key=self.api_key)
            else:
                wandb.login()
            
            self._authenticated = True
            logger.info("Successfully authenticated with W&B")
            return True
            
        except Exception as e:
            logger.error(f"W&B authentication failed: {str(e)}")
            self._authenticated = False
            return False
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self._authenticated
    
    def init_run(self, entity: str = "SPEAK-ASR-uom", project: str = "whisper-finetune-sinhala"):
        """Initialize a W&B run."""
        return wandb.init(entity=entity, project=project)
    
    @staticmethod
    def get_api_key_from_env() -> Optional[str]:
        """Retrieve W&B API key from environment variable."""
        return os.getenv('WANDB_API_KEY')