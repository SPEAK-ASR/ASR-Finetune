# Experiment 9

**Date:** 2026-02-21
**Branch:** `experiment-9`
**Follow-up from:** Experiment 8
**Model push target:** `SPEAK-ASR/whisper-si-exp-9` _(note: `HF_MODEL_ID` in `training.py` still points to `exp-8` — update before training)_

---

## Objective

Evaluate the effect of switching from the raw OpenSLR dataset (short clips, 2–6 sec, used in Experiment 8) to a duration-normalized, noise-removed version (5–25 sec, near-normal distribution, constructed by concatenating clips from the same speaker).

The goal is to determine whether longer, context-rich utterances and cleaner audio improve WER — testing whether data preprocessing quality is a meaningful lever beyond model/LoRA tuning.

---

## Key Change

|                    | Experiment 8                                 | Experiment 9                                                |
| ------------------ | -------------------------------------------- | ----------------------------------------------------------- |
| **Dataset**        | `SPEAK-ASR/openslr-sinhala-asr-preprocessed` | `SPEAK-ASR/openslr-sinhala-asr-norm-noise-rem-preprocessed` |
| **Audio duration** | ~2–6 sec (short clips, original OpenSLR)     | ~5–25 sec (near-normal distribution)                        |
| **Construction**   | Raw original utterances, no concatenation    | Clips concatenated from same speaker to normalize duration  |
| **Noise removal**  | No                                           | Yes (background noise removed)                              |

---

## Configuration

### Model

| Parameter        | Value                  |
| ---------------- | ---------------------- |
| Base model       | `openai/whisper-small` |
| Language         | Sinhala (`si`)         |
| Task             | Transcribe             |
| Max token length | 1024                   |

### LoRA

Best params from Optuna trial #18 (best WER: 37.13% on Experiment 8 dataset).

| Parameter        | Value              |
| ---------------- | ------------------ |
| `r`              | 21                 |
| `lora_alpha`     | 250                |
| `lora_dropout`   | 0.0037             |
| `target_modules` | `q_proj`, `v_proj` |
| `bias`           | none               |

### Training

| Parameter                   | Value            |
| --------------------------- | ---------------- |
| Learning rate               | 3e-5             |
| Epochs                      | 5                |
| Per-device train batch size | 32               |
| Per-device eval batch size  | 256              |
| Gradient accumulation steps | 1                |
| Warmup steps                | 200              |
| Eval & save steps           | 1500             |
| LR scheduler                | linear (default) |

### Precision

| Parameter        | Value |
| ---------------- | ----- |
| `bf16`           | True  |
| `bf16_full_eval` | True  |
| `tf32`           | True  |

---

## Hypothesis

Training on longer, cleaner utterances should give the model more acoustic context per sample and reduce noise interference. If Experiment 9 WER improves over Experiment 8, it validates that duration normalization and background noise removal are meaningful data engineering improvements beyond LoRA hyperparameter tuning.

---

## Results

| Metric               | Value            |
| -------------------- | ---------------- |
| Eval WER             | _(to be filled)_ |
| Best checkpoint step | _(to be filled)_ |
| Training time        | _(to be filled)_ |

---

## Notes

- `torch_compile` was tested in Experiment 8 but reverted — the dynamic label lengths from batch padding caused constant recompilations and negated any speedup.
- NCCL env var updated in `start.sh`: `NCCL_ASYNC_ERROR_HANDLING` → `TORCH_NCCL_ASYNC_ERROR_HANDLING`.
