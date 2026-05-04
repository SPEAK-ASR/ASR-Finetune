"""Configuration for the joint ASR + post-processor pipeline (Stage 2)."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for joint ASR + post-processing training.

    Joint loss: L = beta_post * CE(postproc, clean) + alpha_asr * CE(whisper, asr_hyp).
    alpha_asr keeps Whisper grounded (prevents drift); beta_post is the primary objective.
    """

    # Loss weights
    alpha_asr: float = 0.2
    beta_post: float = 1.0

    # Whisper encoder is usually frozen in Stage 2: acoustics are already solid
    # from the standalone Whisper fine-tune, and this saves ~40% VRAM.
    freeze_whisper_encoder: bool = True

    # Whether to apply LoRA to Whisper's decoder during joint fine-tuning.
    # If False, only the projection layer + post-processor are trained on the Whisper side.
    apply_lora_to_whisper: bool = True

    # Projection layer settings (Whisper hidden dim -> Post-processor hidden dim)
    proj_dropout: float = 0.1
    proj_layer_norm: bool = True

    # How often to refresh the cached ASR hypotheses used for teacher-forcing the
    # Whisper decoder. 0 disables refreshing (keep static hyps from Stage 0).
    # Refreshing mitigates exposure bias as Whisper LoRA updates drift the true
    # error distribution away from the one stored in Stage 0.
    refresh_hyp_every_n_epochs: int = 0

    # Starting checkpoint for the Whisper side in Stage 2. If None, uses
    # ModelConfig.base_model_name (i.e. the original Whisper, not yours).
    # Typically set to your fine-tuned checkpoint from Exp-10.
    whisper_warmstart_repo: str = "SPEAK-ASR/whisper-si-exp-10-medium"

    # HF Hub repo for the final joint checkpoint pushed at end of training.
    hub_joint_repo: str = "SPEAK-ASR/whisper-si-joint-v1"


_PIPELINE_CONFIG = PipelineConfig()


def get_pipeline_config() -> PipelineConfig:
    """Get the pipeline configuration instance."""
    return _PIPELINE_CONFIG
