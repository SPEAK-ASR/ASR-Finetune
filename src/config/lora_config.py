from peft import LoraConfig as PEFTLoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional


class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training."""

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "SEQ_2_SEQ_LM",
    ):
        """
        Initialize LoRA configuration.

        Args:
            r: LoRA attention dimension (rank). Higher values = more trainable parameters
            lora_alpha: LoRA scaling parameter. Typically set to 2*r or r
            target_modules: List of module names to apply LoRA to. 
                           For Whisper: ["q_proj", "v_proj"] or ["q_proj", "v_proj", "k_proj", "o_proj"]
                           None will auto-select appropriate modules
            lora_dropout: Dropout probability for LoRA layers
            bias: Bias training strategy. Options: "none", "all", "lora_only"
            task_type: Type of task. Use "SEQ_2_SEQ_LM" for Whisper
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type

    def to_peft_config(self) -> PEFTLoraConfig:
        """
        Convert to PEFT LoraConfig.

        Returns:
            PEFTLoraConfig object
        """
        return PEFTLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
        )


def apply_lora_to_model(model, lora_config: LoRAConfig):
    """
    Apply LoRA to the model for parameter-efficient fine-tuning.

    Args:
        model: The base model to apply LoRA to
        lora_config: LoRA configuration

    Returns:
        Model with LoRA applied
    """
    # Prepare model for training (freezes base model parameters)
    model = prepare_model_for_kbit_training(model)
    
    # Get PEFT config and apply LoRA
    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model
