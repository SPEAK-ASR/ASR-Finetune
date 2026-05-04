"""TrainerCallback that periodically re-runs Whisper inference over the training
dataset to refresh the cached ASR hypothesis tokens used for teacher forcing.

Mitigates exposure bias: as Whisper's LoRA weights drift during joint training,
the Stage-0 hypotheses become stale and the post-processor stops seeing the
model's current error distribution. Refreshing every N epochs restores the match.

Disabled (no-op) when ``CONFIG.pipeline.refresh_hyp_every_n_epochs <= 0``.
"""

from typing import Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerControl, TrainerState

from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RefreshHypothesesCallback(TrainerCallback):
    """Re-runs ``model.whisper.generate`` over ``train_dataset`` every N epochs and
    rewrites the ``asr_hyp_labels`` column with the new hypotheses.

    Assumes the training dataset is a HuggingFace ``datasets.Dataset`` (so we can
    mutate the column). Works on the main process only; other ranks no-op and
    rely on dataset broadcasting at the start of the next epoch.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        whisper_tokenizer,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        self.train_dataset = train_dataset
        self.whisper_tokenizer = whisper_tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def on_epoch_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        every = CONFIG.pipeline.refresh_hyp_every_n_epochs
        if every is None or every <= 0:
            return control

        if int(state.epoch) % every != 0:
            return control

        # Only refresh on main process
        if state.is_local_process_zero is False:
            return control

        model = kwargs.get("model")
        if model is None:
            logger.warning("RefreshHypothesesCallback: no model in callback kwargs; skipping")
            return control

        underlying = model.module if hasattr(model, "module") else model
        whisper = underlying.whisper
        device = next(whisper.parameters()).device
        whisper.eval()

        logger.info(
            f"[RefreshHypothesesCallback] Regenerating ASR hypotheses at epoch {state.epoch}..."
        )

        def _collate(batch):
            feats = torch.stack([torch.as_tensor(x["input_features"]) for x in batch])
            return {"input_features": feats}

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
        )

        new_hyps: list[list[int]] = []
        with torch.no_grad():
            for batch in loader:
                input_features = batch["input_features"].to(device)
                gen = whisper.generate(
                    input_features=input_features,
                    max_new_tokens=CONFIG.model.max_token_length,
                    num_beams=1,
                )
                for ids in gen.cpu().tolist():
                    new_hyps.append(ids)

        if len(new_hyps) != len(self.train_dataset):
            logger.warning(
                f"[RefreshHypothesesCallback] got {len(new_hyps)} hyps vs {len(self.train_dataset)} samples; skipping update"
            )
            whisper.train()
            return control

        # Rewrite the column in-place.
        self.train_dataset = self.train_dataset.remove_columns(["asr_hyp_labels"]).add_column(
            "asr_hyp_labels", new_hyps
        )
        # The Trainer keeps a reference to the original dataset; we need to
        # mutate in place. HF datasets are immutable-ish, so we update via
        # the trainer object.
        trainer = kwargs.get("trainer") or kwargs.get("tr") or None
        if trainer is not None:
            trainer.train_dataset = self.train_dataset

        whisper.train()
        logger.info(
            f"[RefreshHypothesesCallback] Refreshed {len(new_hyps)} hypotheses"
        )
        return control
