from dataclasses import dataclass, field

from typing import List

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation).

    Best params from Optuna trial #18 (WER: 37.13):
        r=21, lora_alpha=250, lora_dropout=0.0037
    """
    r: int = 101
    lora_alpha: int = 144
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.0884583610934014
    bias: str = "none"

_LORA_CONFIG = LoRAConfig()

def get_lora_config() -> LoRAConfig:
    """Get the LoRA configuration instance."""
    return _LORA_CONFIG
