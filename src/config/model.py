from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    
    # Model selection
    base_model_name: str = "openai/whisper-medium"
    language: str = "Sinhala"
    task: str = "transcribe"
    max_token_length: int = 1024

_MODEL_CONFIG = ModelConfig()

def get_model_config() -> ModelConfig:
    """Get the model configuration instance."""
    return _MODEL_CONFIG

def get_base_model_name() -> str:
    """Get the base model name from the configuration."""
    return _MODEL_CONFIG.base_model_name