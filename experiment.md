# Experiment 10

**Date:** 2026-02-23
**Branch:** `experiment-10`
**Follow-up from:** Experiment 9
**Model push target:** `SPEAK-ASR/whisper-si-exp-10`

---

## Objective

Evaluate the effect of new LoRA hyperparameters (from a fresh Optuna search on the Experiment 9 dataset) combined with a significantly larger batch size and longer training schedule. The dataset and model architecture remain unchanged from Experiment 9.

The goal is to determine whether re-tuning LoRA parameters specifically for the norm/noise-removed dataset — rather than reusing parameters optimised on raw short clips — yields a further WER reduction.

---

## Key Change

|                      | Experiment 9                                                | Experiment 10                                               |
| -------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| **Dataset**          | `SPEAK-ASR/openslr-sinhala-asr-norm-noise-rem-preprocessed` | `SPEAK-ASR/openslr-sinhala-asr-norm-noise-rem-preprocessed` |
| **LoRA `r`**         | 21                                                          | 101                                                         |
| **LoRA `alpha`**     | 250                                                         | 144                                                         |
| **LoRA dropout**     | 0.0037                                                      | 0.0885                                                      |
| **Train batch size** | 32                                                          | 128                                                         |
| **Epochs**           | 5                                                           | 20                                                          |
| **Eval & save**      | Every 1500 steps                                            | Every epoch                                                 |
| **Learning rate**    | 3e-5                                                        | 6.926e-4                                                    |
| **Warmup steps**     | 200                                                         | 295                                                         |

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

New Optuna trial on the Experiment 9 (norm/noise-removed) dataset.

| Parameter        | Value              |
| ---------------- | ------------------ |
| `r`              | 101                |
| `lora_alpha`     | 144                |
| `lora_dropout`   | 0.0885             |
| `target_modules` | `q_proj`, `v_proj` |
| `bias`           | none               |

### Training

| Parameter                   | Value    |
| --------------------------- | -------- |
| Learning rate               | 6.926e-4 |
| Epochs                      | 20       |
| Per-device train batch size | 128      |
| Per-device eval batch size  | 256      |
| Gradient accumulation steps | 1        |
| Warmup steps                | 295      |
| Eval & save strategy        | epoch    |
| LR scheduler                | linear   |

### Precision

| Parameter        | Value |
| ---------------- | ----- |
| `bf16`           | True  |
| `bf16_full_eval` | True  |

---

## Hypothesis

Re-running Optuna on the cleaner, longer-clip dataset should yield LoRA hyperparameters better suited to that distribution. Combined with a larger batch size (better gradient estimates) and more epochs, WER should improve over Experiment 9's result on the same dataset.

---

## Results

| Metric                | Value            |
| --------------------- | ---------------- |
| Eval WER              | _(to be filled)_ |
| Best checkpoint epoch | _(to be filled)_ |
| Training time         | _(to be filled)_ |

---

## Notes

- `torch_compile` was tested in Experiment 8 but reverted — the dynamic label lengths from batch padding caused constant recompilations and negated any speedup.
- NCCL env var updated in `start.sh`: `NCCL_ASYNC_ERROR_HANDLING` → `TORCH_NCCL_ASYNC_ERROR_HANDLING`.
- Eval and save switched from step-based to epoch-based to align with the longer training schedule.
