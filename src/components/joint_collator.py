"""Dual-tokenizer data collator for the joint ASR + Post-Processor pipeline.

Handles three streams with different padding rules:
  - input_features: padded by the Whisper feature extractor (fixed-length mel)
  - asr_hyp_labels: Whisper tokenizer pad + replace pad with -100 for CE
  - clean_labels: ByT5 tokenizer pad + replace pad with -100 for CE
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase, WhisperFeatureExtractor

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class JointDataCollator:
    """Collator for the joint pipeline.

    Expects each feature dict to contain:
        - "input_features": list[float]
        - "asr_hyp_labels": list[int]  (Whisper token ids)
        - "clean_labels": list[int]    (ByT5 byte ids)
    """

    feature_extractor: WhisperFeatureExtractor
    whisper_tokenizer: PreTrainedTokenizerBase
    postproc_tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Audio features
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.feature_extractor.pad(input_feats, return_tensors="pt")

        # Whisper-side labels (ASR hypothesis tokens)
        asr_label_features = [{"input_ids": f["asr_hyp_labels"]} for f in features]
        asr_batch = self.whisper_tokenizer.pad(asr_label_features, return_tensors="pt")
        asr_labels = asr_batch["input_ids"].masked_fill(
            asr_batch.attention_mask.ne(1), -100
        )

        # Whisper's tokenizer pre-pends <|startoftranscript|> (decoder_start_token_id);
        # strip it because the joint model shifts labels right internally.
        if (asr_labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            asr_labels = asr_labels[:, 1:]

        batch["asr_hyp_labels"] = asr_labels
        # keep an explicit mask for the projection -> post-proc encoder attention
        batch["asr_hyp_attention_mask"] = (asr_labels != -100).long()

        # Post-processor-side labels (clean text, ByT5 byte ids)
        clean_label_features = [{"input_ids": f["clean_labels"]} for f in features]
        clean_batch = self.postproc_tokenizer.pad(clean_label_features, return_tensors="pt")
        clean_labels = clean_batch["input_ids"].masked_fill(
            clean_batch.attention_mask.ne(1), -100
        )
        batch["clean_labels"] = clean_labels

        return batch
