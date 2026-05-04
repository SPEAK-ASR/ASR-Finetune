"""Joint ASR + Post-Processor model with hidden-state coupling.

Architecture (training):

    input_features ---[Whisper encoder]---> encoder_hidden
    asr_hyp_labels ---+                                     (teacher forcing)
                       \\
                        \\---[Whisper decoder + LoRA]---> dec_hidden [B, T, D_w]
                                                         \\
                                                          \\---[lm_head] ---> asr_logits (aux loss vs hyp)
                                                          |
                                                          V
                                              [Projection 1024 -> D_post]
                                                          |
                                                          V
                                          [ByT5 encoder via inputs_embeds]
                                                          |
                                                          V
                                    [ByT5 decoder w/ clean_labels] ---> post_logits (main loss vs clean)

    L = beta_post * CE(post, clean) + alpha_asr * CE(asr, hyp)

Architecture (inference):

    input_features ---[Whisper.generate]---> asr_token_ids
    Replay through decoder w/ output_hidden_states -> dec_hidden
    Projection -> ByT5.generate(encoder_outputs=projected)

The Whisper decoder is teacher-forced with its own hypothesis tokens (from Stage 0)
so the hidden states reflect what Whisper does when it is *wrong*, which is exactly
what the post-processor must learn to correct.
"""

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    WhisperForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Projection(nn.Module):
    """Small projection block: Linear -> (LayerNorm) -> Dropout.

    Maps Whisper decoder hidden states (D_whisper) into the post-processor's
    encoder-input space (D_post).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_layer_norm: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.layer_norm(self.linear(x)))


class JointASRPostProcessorModel(nn.Module):
    """Joint model combining Whisper + Projection + Post-processor (ByT5).

    We inherit from ``nn.Module`` (not ``PreTrainedModel``) on purpose: the wrapper
    holds two heterogeneous HF models and the PreTrainedModel init tries to pick
    an attention implementation for the wrapper class itself, which doesn't apply
    here. We expose the small surface the HF Trainer needs (``main_input_name``,
    ``config``, ``generation_config``, ``save_pretrained``) manually.
    """

    # Mirror Whisper's main_input_name so HF Trainer treats ``input_features`` as
    # the primary input (matches Seq2SeqTrainer expectations).
    main_input_name = "input_features"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        whisper: WhisperForConditionalGeneration,
        postprocessor: PreTrainedModel,
        whisper_tokenizer: PreTrainedTokenizerBase,
        postproc_tokenizer: PreTrainedTokenizerBase,
    ):
        super().__init__()

        self.whisper = whisper
        self.postprocessor = postprocessor
        self.whisper_tokenizer = whisper_tokenizer
        self.postproc_tokenizer = postproc_tokenizer

        # Expose a ``config`` that the Trainer can poke at (it reads a few fields
        # like ``config.use_cache`` and checks label names via ``config.keys_to_ignore_at_inference``).
        self.config = whisper.config
        # Expose a ``generation_config`` (Trainer checks ``model.generation_config``
        # when predict_with_generate=True).
        self.generation_config = whisper.generation_config

        whisper_d = whisper.config.d_model
        post_d = postprocessor.config.d_model

        self.projection = Projection(
            in_dim=whisper_d,
            out_dim=post_d,
            dropout=CONFIG.pipeline.proj_dropout,
            use_layer_norm=CONFIG.pipeline.proj_layer_norm,
        )

        # Loss weights as buffers so they can be moved with .to(device) and
        # optionally swapped from a callback later.
        self.register_buffer(
            "alpha_asr",
            torch.tensor(float(CONFIG.pipeline.alpha_asr)),
            persistent=False,
        )
        self.register_buffer(
            "beta_post",
            torch.tensor(float(CONFIG.pipeline.beta_post)),
            persistent=False,
        )

        if CONFIG.pipeline.freeze_whisper_encoder:
            logger.info("Freezing Whisper encoder parameters")
            for p in self.whisper.get_encoder().parameters():
                p.requires_grad = False

        logger.info(
            f"JointASRPostProcessorModel built: whisper_d={whisper_d}, "
            f"post_d={post_d}, alpha_asr={CONFIG.pipeline.alpha_asr}, "
            f"beta_post={CONFIG.pipeline.beta_post}"
        )

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------

    def _shift_right_whisper(self, asr_hyp_labels: torch.Tensor) -> torch.Tensor:
        """Build decoder_input_ids for Whisper from labels (shift right by one,
        prepending decoder_start_token_id). Copies the logic from
        ``modeling_whisper.shift_tokens_right`` but local so we can safely feed
        our padding value (-100)."""
        decoder_start_token_id = self.whisper.config.decoder_start_token_id
        pad_token_id = self.whisper.config.pad_token_id

        shifted = asr_hyp_labels.new_zeros(asr_hyp_labels.shape)
        shifted[..., 1:] = asr_hyp_labels[..., :-1].clone()
        shifted[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            pad_token_id = 0
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def _whisper_decoder_pass(
        self,
        input_features: torch.Tensor,
        asr_hyp_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run Whisper encoder + decoder with teacher-forcing on the ASR hypothesis.

        Returns a dict with ``asr_logits`` [B,T,V_w] and ``dec_hidden`` [B,T,D_w].
        """
        decoder_input_ids = self._shift_right_whisper(asr_hyp_labels)

        out = self.whisper(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        dec_hidden = out.decoder_hidden_states[-1]
        return {"asr_logits": out.logits, "dec_hidden": dec_hidden}

    def _postproc_forward(
        self,
        projected: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        clean_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Run the post-processor with a supplied encoder_outputs (projected Whisper
        hidden states) and labels. Returns post_logits [B, T_out, V_post]."""
        encoder_outputs = BaseModelOutput(last_hidden_state=projected)
        post_out = self.postprocessor(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=clean_labels,
            return_dict=True,
        )
        return post_out.logits

    # ------------------------------------------------------------------
    # HF Trainer interface
    # ------------------------------------------------------------------

    def forward(
        self,
        input_features: torch.Tensor,
        asr_hyp_labels: torch.Tensor,
        clean_labels: torch.Tensor,
        asr_hyp_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Training forward pass computing the joint loss.

        Args:
            input_features: [B, n_mels, T_mel] log-Mel features for Whisper
            asr_hyp_labels: [B, T_hyp] Whisper token ids (padded with -100) - the
                ASR's hypothesised transcript, used both as aux CE target and for
                teacher-forcing Whisper's decoder.
            clean_labels: [B, T_clean] ByT5 token ids (padded with -100) - the
                main CE target for the post-processor.
            asr_hyp_attention_mask: [B, T_hyp] optional 1/0 mask for ASR
                hypothesis tokens; if None, derived from ``asr_hyp_labels != -100``.

        Returns a Seq2SeqLMOutput whose ``loss`` is the weighted joint loss and
        whose ``logits`` are the post-processor's logits (so Trainer metrics can
        decode them).
        """
        whisper_out = self._whisper_decoder_pass(input_features, asr_hyp_labels)
        dec_hidden = whisper_out["dec_hidden"]
        asr_logits = whisper_out["asr_logits"]

        if asr_hyp_attention_mask is None:
            asr_hyp_attention_mask = (asr_hyp_labels != -100).long()

        projected = self.projection(dec_hidden)

        post_logits = self._postproc_forward(
            projected=projected,
            encoder_attention_mask=asr_hyp_attention_mask,
            clean_labels=clean_labels,
        )

        asr_loss = F.cross_entropy(
            asr_logits.reshape(-1, asr_logits.size(-1)),
            asr_hyp_labels.reshape(-1),
            ignore_index=-100,
        )
        post_loss = F.cross_entropy(
            post_logits.reshape(-1, post_logits.size(-1)),
            clean_labels.reshape(-1),
            ignore_index=-100,
        )

        loss = self.beta_post * post_loss + self.alpha_asr * asr_loss

        return Seq2SeqLMOutput(
            loss=loss,
            logits=post_logits,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        """End-to-end generation: audio -> Whisper decode -> projection -> ByT5 decode.

        Returns ByT5 token ids of the final clean transcript.
        """
        # Step 1: Whisper generates its best hypothesis.
        # Cap at the model's own max_target_positions (448 for whisper-tiny,
        # 448 for medium too) minus headroom for the prompt tokens Whisper adds.
        whisper_max_pos = getattr(
            self.whisper.config, "max_target_positions", 448
        )
        whisper_max_new = min(
            CONFIG.model.max_token_length, max(16, whisper_max_pos - 8)
        )
        whisper_ids = self.whisper.generate(
            input_features=input_features,
            max_new_tokens=whisper_max_new,
            num_beams=num_beams,
        )

        # Step 2: Replay the hypothesis through the decoder to get hidden states.
        # (Generation returns tokens starting with decoder_start_token_id; we feed
        # that whole sequence as decoder_input_ids.)
        replay = self.whisper(
            input_features=input_features,
            decoder_input_ids=whisper_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        dec_hidden = replay.decoder_hidden_states[-1]

        # Step 3: Project + post-processor generation.
        projected = self.projection(dec_hidden)
        encoder_attention_mask = (
            whisper_ids != self.whisper.config.pad_token_id
        ).long() if self.whisper.config.pad_token_id is not None else torch.ones(
            whisper_ids.shape, dtype=torch.long, device=whisper_ids.device
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=projected)
        post_ids = self.postprocessor.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **gen_kwargs,
        )
        return post_ids

    # ------------------------------------------------------------------
    # Saving helpers (save the three sub-modules to sub-directories)
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        import os

        os.makedirs(save_directory, exist_ok=True)
        self.whisper.save_pretrained(os.path.join(save_directory, "whisper"), **kwargs)
        self.postprocessor.save_pretrained(
            os.path.join(save_directory, "postprocessor"), **kwargs
        )
        torch.save(
            self.projection.state_dict(),
            os.path.join(save_directory, "projection.pt"),
        )
        logger.info(f"Joint model saved to {save_directory}")
