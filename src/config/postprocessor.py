"""Configuration for the post-processing (text polishing) model used in the joint
ASR + post-processing pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PostProcessorConfig:
    """Configuration for the post-processing seq2seq model.

    ByT5-small is the default because its byte-level tokenizer handles Sinhala
    (pillams / zwj / zwnj / rare characters) without OOV issues, and because the
    model's mC4 pretraining already covers Sinhala.
    """

    # HF model id for the base post-processor architecture + tokenizer
    model_name: str = "google/byt5-small"

    # Hidden size of the post-processor's encoder (ByT5-small: 1472, base: 1536, mt5-small: 512)
    # Only used to size the projection from Whisper decoder hidden states.
    hidden_dim: int = 1472

    # If set, load post-processor weights from this checkpoint instead of model_name
    # (used to warm-start the joint training from Stage 1's output).
    warmstart_path: Optional[str] = None

    # Whether to freeze the post-processor's encoder during joint training.
    # Usually False: we want it to adapt to Whisper hidden-state inputs.
    freeze_encoder: bool = False

    # Max token length for the post-processor's decoder (clean-text labels).
    # ByT5 works at byte level so this should be ~4x word count.
    max_target_length: int = 1024

    # HF Hub repo for the Stage 1 warm-started post-processor checkpoint.
    hub_warmstart_repo: str = "SPEAK-ASR/byt5-sinhala-post-pretrain"


_POSTPROCESSOR_CONFIG = PostProcessorConfig()


def get_postprocessor_config() -> PostProcessorConfig:
    """Get the post-processor configuration instance."""
    return _POSTPROCESSOR_CONFIG
