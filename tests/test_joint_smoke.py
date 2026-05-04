"""Smoke test for the joint ASR + Post-Processor pipeline.

Verifies (with tiny weights and synthetic data) that:
  - the joint model forward pass runs end-to-end
  - the joint loss combines both streams and has a gradient
  - the custom generate() chains Whisper decode -> projection -> ByT5 decode
  - the dual-tokenizer collator produces correctly shaped tensors

Does NOT train meaningfully or check quality; it's a structural/shape test.

Run locally with:

    python -m tests.test_joint_smoke

Does not require network access if the tiny models are already cached.
"""

import os
import sys

# Ensure project root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.config.config import CONFIG
from src.components.joint_model import JointASRPostProcessorModel
from src.components.joint_collator import JointDataCollator


TINY_WHISPER = "openai/whisper-tiny"
TINY_BYT5 = "google/byt5-small"


def _build_joint_model():
    whisper_tok = WhisperTokenizer.from_pretrained(
        TINY_WHISPER, language="english", task="transcribe"
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(TINY_WHISPER)
    whisper = WhisperForConditionalGeneration.from_pretrained(TINY_WHISPER)
    whisper.generation_config.language = "english"
    whisper.generation_config.task = "transcribe"
    whisper.generation_config.forced_decoder_ids = None

    byt5_tok = AutoTokenizer.from_pretrained(TINY_BYT5)
    byt5 = AutoModelForSeq2SeqLM.from_pretrained(TINY_BYT5)

    # Override pipeline config for smoke test: don't freeze encoder (tiny model),
    # so we can actually check gradient flow into Whisper.
    CONFIG.pipeline.freeze_whisper_encoder = False
    CONFIG.pipeline.proj_dropout = 0.0
    CONFIG.pipeline.proj_layer_norm = True
    CONFIG.pipeline.alpha_asr = 0.3
    CONFIG.pipeline.beta_post = 1.0

    model = JointASRPostProcessorModel(
        whisper=whisper,
        postprocessor=byt5,
        whisper_tokenizer=whisper_tok,
        postproc_tokenizer=byt5_tok,
    )
    return model, whisper_tok, byt5_tok, feature_extractor


def _fake_batch(feature_extractor, whisper_tok, byt5_tok, batch_size=2):
    """Build a tiny batch of random audio + two short transcripts."""
    rng = np.random.default_rng(0)
    samples = []
    for i in range(batch_size):
        audio = rng.standard_normal(16000 * 2).astype(np.float32)  # 2s @ 16k
        feats = feature_extractor(audio, sampling_rate=16000).input_features[0]
        hyp_text = f"this is hypothesisz {i}"  # intentionally typo'd
        clean_text = f"this is hypothesis {i}"
        samples.append({
            "input_features": feats,
            "asr_hyp_labels": whisper_tok(hyp_text).input_ids,
            "clean_labels": byt5_tok(clean_text).input_ids,
        })
    return samples


def test_forward_and_backward():
    print("Loading tiny joint model...")
    model, w_tok, b_tok, fe = _build_joint_model()
    model.train()

    collator = JointDataCollator(
        feature_extractor=fe,
        whisper_tokenizer=w_tok,
        postproc_tokenizer=b_tok,
        decoder_start_token_id=model.whisper.config.decoder_start_token_id,
    )
    batch = collator(_fake_batch(fe, w_tok, b_tok, batch_size=2))

    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")

    assert batch["input_features"].ndim == 3
    assert batch["asr_hyp_labels"].ndim == 2
    assert batch["clean_labels"].ndim == 2

    out = model(
        input_features=batch["input_features"],
        asr_hyp_labels=batch["asr_hyp_labels"],
        clean_labels=batch["clean_labels"],
        asr_hyp_attention_mask=batch["asr_hyp_attention_mask"],
    )
    print(f"Joint loss: {out.loss.item():.4f}  |  logits: {tuple(out.logits.shape)}")
    assert torch.isfinite(out.loss), "loss is non-finite"

    out.loss.backward()

    # Check that gradients flow to all three sub-components.
    proj_grad = next(p for p in model.projection.parameters() if p.grad is not None)
    post_grad = next(p for p in model.postprocessor.parameters() if p.grad is not None)
    whisper_grad = next(
        (p for p in model.whisper.parameters() if p.grad is not None and p.grad.abs().sum() > 0),
        None,
    )

    assert proj_grad is not None, "projection has no gradient"
    assert post_grad is not None, "postprocessor has no gradient"
    assert whisper_grad is not None, "whisper has no gradient"
    print("Gradients confirmed in projection, post-processor, and whisper.")


def test_generate():
    print("Running joint generate() smoke...")
    model, w_tok, b_tok, fe = _build_joint_model()
    model.eval()

    rng = np.random.default_rng(1)
    audio = rng.standard_normal(16000 * 2).astype(np.float32)
    feats = fe(audio, sampling_rate=16000, return_tensors="pt").input_features

    out_ids = model.generate(feats, max_new_tokens=16, num_beams=1)
    decoded = b_tok.batch_decode(out_ids, skip_special_tokens=True)
    print(f"  generated ids shape: {tuple(out_ids.shape)}")
    print(f"  decoded: {decoded}")
    assert out_ids.ndim == 2


if __name__ == "__main__":
    test_forward_and_backward()
    test_generate()
    print("\nAll smoke checks passed.")
