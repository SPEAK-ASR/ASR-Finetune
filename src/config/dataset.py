from dataclasses import dataclass, field
from typing import Literal, List

@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset source."""
    dataset_name: str
    train_split: str | None = "train"
    test_split: str | None = "test"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    # Multiple datasets configuration
    # Each dataset has: dataset_name, train_split, test_split
    # All datasets will be combined into a single DatasetDict
    datasets: List[SingleDatasetConfig] = field(default_factory=lambda: [
        SingleDatasetConfig(
            dataset_name="SPEAK-ASR/openslr-sinhala-asr-preprocessed-1",
            train_split="train",
            test_split="test"
        ),
        SingleDatasetConfig(
            dataset_name="SPEAK-ASR/openslr-sinhala-asr-preprocessed-2",
            train_split="train",
            test_split=None
        ),
        SingleDatasetConfig(
            dataset_name="SPEAK-ASR/openslr-sinhala-asr-preprocessed-3",
            train_split="train",
            test_split=None
        ),
        SingleDatasetConfig(
            dataset_name="SPEAK-ASR/youtube-sinhala-asr-preprocessed",
            train_split="train",
            test_split="test"
        ),
    ])

    use_auth_token: bool = True
    keep_in_memory: bool = False  # Whether to load datasets into memory
    streaming: bool = False  # Whether to use dataset streaming to reduce loading time

    # Audio preprocessing
    sample_rate: int = 16000
    audio_column: str = "audio"
    transcript_column: str = "text"

_DATASET_CONFIG = DatasetConfig()

def get_dataset_config() -> DatasetConfig:
    """Get the dataset configuration instance."""
    return _DATASET_CONFIG

def get_dataset_names() -> List[str]:
    """Get a list of dataset names from the configuration."""
    return [ds.dataset_name for ds in _DATASET_CONFIG.datasets]