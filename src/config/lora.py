from dataclasses import dataclass, field

from typing import List

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 32
    lora_alpha: int = 64
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"

_LORA_CONFIG = LoRAConfig()

def get_lora_config() -> LoRAConfig:
    """Get the LoRA configuration instance."""
    return _LORA_CONFIG
