"""
HuggingFace Hub integration for Whisper fine-tuning.
Handles authentication and model repository management.
"""

from huggingface_hub import login
from typing import Optional
import os

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HuggingFaceAuthenticator:
    """
    Handles authentication with HuggingFace Hub.
    Manages login and token validation for accessing HuggingFace resources.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the authenticator.
        
        Args:
            token: HuggingFace API token. If None, will prompt for interactive login.
        """
        self.token = token
        self._authenticated = False
        logger.info("HuggingFaceAuthenticator initialized")
    
    def authenticate(self, use_auth_token: bool = True) -> bool:
        """
        Authenticate with HuggingFace Hub.
        
        Args:
            use_auth_token: Whether to use token-based authentication
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            logger.info("Attempting to authenticate with HuggingFace Hub...")
            
            if self.token:
                logger.debug("Using provided token for authentication")
                login(token=self.token)
            else:
                logger.debug("Starting interactive login process")
                login()
            
            self._authenticated = True
            logger.info("Successfully authenticated with HuggingFace Hub")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            self._authenticated = False
            return False
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self._authenticated
    
    @staticmethod
    def get_token_from_env() -> Optional[str]:
        """
        Retrieve HuggingFace token from environment variable.
        
        Returns:
            Token string if found, None otherwise
        """
        token = os.getenv('HF_TOKEN')
        if token:
            logger.debug("Token found in environment variables")
        else:
            logger.debug("No token found in environment variables")
        return token
