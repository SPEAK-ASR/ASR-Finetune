from dataclasses import dataclass, field
from typing import List
from src.config.training import HF_MODEL_ID
from src.config.dataset import get_dataset_names
from src.config.model import get_base_model_name


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace Hub integration."""
    
    # Authentication
    dataset_tags: List[str] = field(default_factory=get_dataset_names)
    dataset: str = " | ".join(get_dataset_names())
    language: str = "si"
    dataset_args: str = "config: si, split: test"
    model_name: str = HF_MODEL_ID
    finetuned_from: str = get_base_model_name()
    tasks: str = "automatic-speech-recognition"

_HUGGINGFACE_CONFIG = HuggingFaceConfig()

def get_huggingface_config() -> HuggingFaceConfig:
    """Get the HuggingFace configuration instance."""
    return _HUGGINGFACE_CONFIG