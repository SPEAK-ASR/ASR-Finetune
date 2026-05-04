"""Trainer for the joint ASR + Post-Processor pipeline.

Wraps HuggingFace's Seq2SeqTrainer but overrides ``prediction_step`` so that
``predict_with_generate=True`` routes through the joint model's custom
``generate(input_features, ...)`` (which chains Whisper decode -> projection ->
post-processor decode).

Also applies LoRA to the Whisper decoder (via PEFT) if configured.
"""

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
)
from peft import LoraConfig as PEFTLoraConfig, get_peft_model

from src.config.config import CONFIG
from src.components.joint_model import JointASRPostProcessorModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class _JointSeq2SeqTrainer(Seq2SeqTrainer):
    """Subclass that calls ``model.generate(input_features=...)`` during eval."""

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # If we're not doing generation, fall back to standard loss-only path.
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        inputs = self._prepare_inputs(inputs)

        # Run generation through the joint model.
        gen_kwargs = {
            "max_new_tokens": self.args.generation_max_length or 256,
            "num_beams": self.args.generation_num_beams or 1,
        }

        underlying = model.module if hasattr(model, "module") else model
        generated_tokens = underlying.generate(
            input_features=inputs["input_features"],
            **gen_kwargs,
        )

        # Compute the eval loss via the normal forward (on the same batch) so we
        # still get a scalar loss logged at eval time.
        with torch.no_grad():
            outputs = model(
                input_features=inputs["input_features"],
                asr_hyp_labels=inputs["asr_hyp_labels"],
                clean_labels=inputs["clean_labels"],
                asr_hyp_attention_mask=inputs.get("asr_hyp_attention_mask"),
            )
            loss = outputs.loss.detach()

        # Pad generated_tokens to the max gen length for stacking.
        labels = inputs["clean_labels"]
        return (loss, generated_tokens, labels)


class JointTrainer:
    """Facade wrapper that builds a JointASRPostProcessorModel (with optional LoRA
    on Whisper's decoder) and a Seq2SeqTrainer that can evaluate end-to-end.
    """

    def __init__(
        self,
        joint_model: JointASRPostProcessorModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: Any,
        compute_metrics: Callable,
        postproc_tokenizer: PreTrainedTokenizerBase,
        extra_callbacks: Optional[List[Any]] = None,
    ):
        self.model = joint_model

        # Apply LoRA to the Whisper sub-module if configured.
        if CONFIG.lora is not None and CONFIG.pipeline.apply_lora_to_whisper:
            logger.info("Applying LoRA to Whisper decoder inside joint model")
            peft_config = PEFTLoraConfig(**asdict(CONFIG.lora))
            # Wrap only the Whisper sub-module; the rest (projection, post-proc)
            # stays vanilla. PEFT replaces target modules in-place.
            self.model.whisper = get_peft_model(self.model.whisper, peft_config)
            self.model.whisper.print_trainable_parameters()

        training_kwargs = asdict(CONFIG.training)
        # Stage 2 should push to the joint-pipeline repo, not the ASR-only one.
        training_kwargs["hub_model_id"] = CONFIG.pipeline.hub_joint_repo
        # Joint-training output dir should be distinct so Stage-2 checkpoints
        # don't overwrite Stage-0/1 checkpoints.
        training_kwargs["output_dir"] = f"{training_kwargs['output_dir']}/joint"
        self.training_args = Seq2SeqTrainingArguments(**training_kwargs)

        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=CONFIG.runtime.early_stopping_patience
            )
        ]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        self.trainer = _JointSeq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=postproc_tokenizer,
            callbacks=callbacks,
        )
