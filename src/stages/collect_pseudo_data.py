"""Stage 0: pseudo-data mining.

Run the currently fine-tuned Whisper model over the ASR dataset and store each
sample's ASR hypothesis alongside the gold transcript. The resulting dataset is
used:
  - in Stage 1 to pretrain the post-processor on ASR-error-matched pairs, and
  - in Stage 2 as training data for the joint pipeline (teacher-forcing the
    Whisper decoder with its own hypothesis tokens).

The output dataset has columns:
  - audio  (preserved from the input)
  - text   (gold / clean transcript, preserved from the input)
  - asr_hyp_text (detokenised Whisper hypothesis)
  - clean_text   (alias of ``text``, for consistency with the parallel corpus)
"""

from typing import Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _generate_hypotheses(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    split: Dataset,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    """Run greedy Whisper inference over ``split`` and return the list of
    decoded hypothesis strings (length == len(split))."""

    model.eval()
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    hyps: list[str] = []

    def _iter_batches():
        buf = []
        for ex in split:
            buf.append(ex)
            if len(buf) == batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    with torch.no_grad():
        for i, batch in enumerate(_iter_batches()):
            audio = [ex["audio"] for ex in batch]
            inputs = feature_extractor(
                [a["array"] for a in audio],
                sampling_rate=audio[0]["sampling_rate"],
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)

            gen = model.generate(
                input_features=input_features,
                max_new_tokens=CONFIG.model.max_token_length,
                num_beams=1,
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            hyps.extend([d.strip() for d in decoded])

            if i % 20 == 0:
                logger.info(f"  batch {i}: cumulative hyps={len(hyps)}")

    return hyps


def collect_pseudo_data(
    dataset: DatasetDict,
    whisper_checkpoint: Optional[str] = None,
    batch_size: int = 16,
    push_to_hub_repo: Optional[str] = None,
    token: Optional[str] = None,
) -> DatasetDict:
    """Run Whisper over all splits of ``dataset`` and attach ASR hypothesis columns.

    Args:
        dataset: DatasetDict with at least an ``audio`` and ``text`` column.
        whisper_checkpoint: HF id or path of the Whisper model to use. Defaults to
            ``CONFIG.pipeline.whisper_warmstart_repo``.
        batch_size: Inference batch size.
        push_to_hub_repo: If provided, push the resulting dataset to the Hub here.
        token: HF auth token for pushing.

    Returns:
        DatasetDict with additional columns: ``asr_hyp_text``, ``clean_text``.
    """
    ckpt = whisper_checkpoint or CONFIG.pipeline.whisper_warmstart_repo
    logger.info(f"Loading Whisper from {ckpt} for pseudo-data collection...")

    processor = WhisperProcessor.from_pretrained(
        ckpt,
        language=CONFIG.model.language,
        task=CONFIG.model.task,
        cache_dir=CONFIG.paths.model_cache_dir,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt,
        cache_dir=CONFIG.paths.model_cache_dir,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out: dict[str, Dataset] = {}
    for split_name, split in dataset.items():
        logger.info(f"Generating hypotheses for split '{split_name}' ({len(split)} samples)...")
        hyps = _generate_hypotheses(model, processor, split, batch_size, device)

        new_split = split.add_column("asr_hyp_text", hyps)
        # Provide a ``clean_text`` alias so the Stage-1 pretraining code can read
        # either the pseudo or parallel dataset with the same column name.
        new_split = new_split.add_column("clean_text", list(new_split["text"]))
        out[split_name] = new_split

    out_dsd = DatasetDict(out)
    logger.info(f"Pseudo dataset ready: { {k: len(v) for k, v in out_dsd.items()} }")

    if push_to_hub_repo is not None:
        logger.info(f"Pushing pseudo dataset to Hub: {push_to_hub_repo}")
        out_dsd.push_to_hub(push_to_hub_repo, token=token, private=False)

    return out_dsd
