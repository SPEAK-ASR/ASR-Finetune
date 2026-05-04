"""Post-processor component: loads the seq2seq text-polishing model (ByT5 by default)
plus its tokenizer.

Used by:
  - Stage 1 (pretrain_postproc.py): trained alone on parallel + pseudo pairs
  - Stage 2 (joint training): wrapped inside JointASRPostProcessorModel
"""

from typing import Optional, Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PostProcessorComponent:
    """Loads + holds the post-processor model and tokenizer."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        warmstart_path: Optional[str] = None,
    ):
        self.model_name = model_name or CONFIG.postprocessor.model_name
        # warmstart_path takes precedence over model_name if provided
        self.warmstart_path = warmstart_path or CONFIG.postprocessor.warmstart_path

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load both model and tokenizer. Tokenizer always comes from base model_name
        (byte-level tokenizer is fixed by architecture), weights come from
        warmstart_path if provided, otherwise model_name.
        """
        logger.info(f"Loading post-processor tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=CONFIG.paths.model_cache_dir,
        )

        weights_src = self.warmstart_path or self.model_name
        logger.info(f"Loading post-processor weights from {weights_src}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            weights_src,
            cache_dir=CONFIG.paths.model_cache_dir,
        )

        if CONFIG.postprocessor.freeze_encoder:
            logger.info("Freezing post-processor encoder")
            for p in self.model.get_encoder().parameters():
                p.requires_grad = False

        logger.info(
            f"Post-processor loaded: arch={self.model.config.model_type}, "
            f"d_model={getattr(self.model.config, 'd_model', None)}, "
            f"vocab_size={self.model.config.vocab_size}"
        )
        return self.model, self.tokenizer

    def get_model(self) -> PreTrainedModel:
        if self.model is None:
            self.load()
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.tokenizer is None:
            self.load()
        return self.tokenizer
