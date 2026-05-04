# Training guide: Whisper ASR and joint post-processing pipeline

This repository supports two training paths:

1. **Standalone Whisper ASR** (original flow): prepare dataset, fine-tune Whisper with LoRA.
2. **Joint ASR + post-processor** (three stages): mine pseudo pairs from your ASR, warm-start a text polisher (ByT5), then fine-tune Whisper and the post-processor together with hidden-state coupling.

Use this guide to configure tasks, datasets, and checkpoints before you launch training.

---

## Prerequisites

- **Python environment**: activate the project venv (for example `.venv`).
- **Hugging Face**: `HF_TOKEN` in the environment or in `.env` (see `.env.example`). You need write access to any Hub repos you push to.
- **Weights & Biases**: `WANDB_API_KEY` in the environment or in `.env`. `main.py` initialises W&B before every task; keep the key set even for stages that log lightly.
- **GPU**: Stage 2 loads Whisper plus ByT5; plan for noticeably more VRAM than Whisper-only training. Reduce `per_device_train_batch_size` in [src/config/training.py](src/config/training.py) if you hit OOM (for example from 128 toward 32).
- **Accelerate** (optional): [start.sh](start.sh) uses `accelerate launch` with [accelerate_config.yaml](accelerate_config.yaml). For a single GPU you can run `python main.py` directly after editing config.

---

## How runs are selected

Training behaviour is controlled by **`CONFIG.runtime.task`** in [src/config/config.py](src/config/config.py).

| `task` value | Loads raw ASR dataset? | Purpose |
|--------------|----------------------|---------|
| `prepare_dataset` | Yes | Build Mel + tokenised labels and push a preprocessed Hub dataset. |
| `finetune_asr_model` | Yes | Fine-tune Whisper only (existing pipeline). |
| `collect_pseudo_data` | Yes | **Stage 0**: run your fine-tuned Whisper on train/test, push pseudo dataset. |
| `pretrain_postproc` | No | **Stage 1**: train ByT5 on parallel + pseudo text pairs. |
| `finetune_joint_pipeline` | No | **Stage 2**: joint Whisper + ByT5 training on the pseudo dataset. |

After changing `task`, run training the same way you usually do (for example `./start.sh start` or `accelerate launch ... main.py`).

---

## Standalone Whisper (unchanged flow)

1. Point [src/config/dataset.py](src/config/dataset.py) `datasets` at your audio Hub dataset (columns: `audio`, `text`).
2. Set `CONFIG.runtime.task` to `prepare_dataset` once if you need a new preprocessed snapshot. The Hub repository name used when pushing is defined in [main.py](main.py) (`_create_prepared_dataset`, variable `repo_name`). Confirm it matches your Hub layout before running.
3. Set `task` to `finetune_asr_model` and tune [src/config/model.py](src/config/model.py), [src/config/lora.py](src/config/lora.py), and [src/config/training.py](src/config/training.py).

---

## Joint pipeline: recommended order

Run stages in order unless you already have artefacts on the Hub.

### Stage 0 — Pseudo-data collection (`collect_pseudo_data`)

**Goal:** For each training (and test) example, store the ASR hypothesis next to the gold transcript so the post-processor sees *this* Whisper’s errors.

**Configure:**

- [src/config/config.py](src/config/config.py): `runtime.task = "collect_pseudo_data"`.
- [src/config/dataset.py](src/config/dataset.py): `datasets` — same raw ASR dataset you want hypotheses for (typically your preprocessed train/test Hub dataset).
- [src/config/pipeline.py](src/config/pipeline.py): `whisper_warmstart_repo` — the Whisper checkpoint used for inference (for example your fine-tuned `SPEAK-ASR/whisper-si-exp-10-medium`).

**Output:** A dataset pushed to `CONFIG.dataset.pseudo_dataset_name` (default `SPEAK-ASR/openslr-sinhala-asr-pseudo-exp10`). Expected columns after the script runs include:

- `audio`, `text` (from the source dataset)
- `asr_hyp_text` — decoded Whisper hypothesis
- `clean_text` — copy of gold `text` (for a consistent column name in later stages)

**Note:** Stage 0 stores string hypotheses. Stage 2 re-tokenises them with the Whisper tokenizer during `prepare_joint_dataset`.

---

### Stage 1 — Post-processor pretraining (`pretrain_postproc`)

**Goal:** Warm-start ByT5 (or your configured seq2seq) on noisy→clean Sinhala text, using both your parallel corpus and Stage 0 pseudo pairs.

**Configure:**

- [src/config/config.py](src/config/config.py): `runtime.task = "pretrain_postproc"`.
- [src/config/dataset.py](src/config/dataset.py):
  - `parallel_text_dataset` — Hub id of your ~50k (or similar) parallel set, or `None` if you only use pseudo data.
  - `parallel_noisy_column` / `parallel_clean_column` — column names in that dataset (defaults: `noisy_text`, `clean_text`).
  - `pseudo_dataset_name` — must match the Hub repo produced in Stage 0 (or leave default if you used it).
- [src/config/postprocessor.py](src/config/postprocessor.py): `model_name` (default `google/byt5-small`), optional `warmstart_path` for a local checkpoint.

**Behaviour:** [src/stages/pretrain_postproc.py](src/stages/pretrain_postproc.py) merges parallel and pseudo splits, renames columns to `noisy_text` / `clean_text`, tokenises with ByT5, trains with `Seq2SeqTrainer`, saves under `checkpoints/postproc_pretrain/final`, and pushes to `CONFIG.postprocessor.hub_warmstart_repo` when `main.py` passes a non-null push target (it uses your configured hub repo).

**If Stage 1 fails on missing data:** Ensure at least one of the parallel dataset or the pseudo dataset loads successfully; otherwise the script raises a clear error.

---

### Stage 2 — Joint fine-tuning (`finetune_joint_pipeline`)

**Goal:** Train Whisper (typically decoder LoRA) + projection + ByT5 with a joint loss: main term on clean text from the post-processor, auxiliary term keeping Whisper aligned with cached hypotheses.

**Configure:**

- [src/config/config.py](src/config/config.py): `runtime.task = "finetune_joint_pipeline"`.
- [src/config/dataset.py](src/config/dataset.py): `pseudo_dataset_name` must point at the Stage 0 dataset on the Hub.
- [src/config/pipeline.py](src/config/pipeline.py):
  - `whisper_warmstart_repo` — Whisper weights + tokenizer (same family as training).
  - `alpha_asr`, `beta_post` — loss weights (defaults 0.2 and 1.0).
  - `freeze_whisper_encoder` — default `True` to save VRAM.
  - `apply_lora_to_whisper` — default `True`; uses [src/config/lora.py](src/config/lora.py) like the ASR trainer.
  - `refresh_hyp_every_n_epochs` — set to a positive integer to enable periodic hypothesis refresh (see callback in [src/components/refresh_hyp_callback.py](src/components/refresh_hyp_callback.py)).
  - `hub_joint_repo` — where Stage 2 checkpoints are pushed when `push_to_hub` is enabled in training config.
- [src/config/postprocessor.py](src/config/postprocessor.py):
  - Set `warmstart_path` to a **local** directory if you want to load Stage 1 from disk instead of the Hub.
  - Otherwise leave `warmstart_path` as `None`; Stage 2 will use `warmstart_path or hub_warmstart_repo` from [main.py](main.py), so the Hub id from Stage 1 is used.

**Eval split:** If the pseudo dataset has only `train`, the code automatically holds out 2% as `test` for evaluation.

**Metrics:** WER is computed on the **post-processor decoded** text (see [src/components/joint_evaluator.py](src/components/joint_evaluator.py)). Generation during eval uses the joint model’s `generate` path (Whisper decode → projection → ByT5 decode).

**Output directory:** [src/components/joint_trainer.py](src/components/joint_trainer.py) sets `output_dir` to `{CONFIG.training.output_dir}/joint` and `hub_model_id` to `CONFIG.pipeline.hub_joint_repo` so joint runs do not overwrite the ASR-only Hub repo in [src/config/training.py](src/config/training.py).

---

## Parallel text dataset schema (Stage 1)

Your parallel Hub dataset should expose two text columns. Defaults in config:

| Role | Default column name |
|------|---------------------|
| Noisy / ASR-like input | `noisy_text` |
| Clean reference | `clean_text` |

If your corpus uses different names, set `parallel_noisy_column` and `parallel_clean_column` in [src/config/dataset.py](src/config/dataset.py).

---

## Smoke test (structure only)

To verify the joint model forward, backward, and `generate` chain without full training:

```bash
.venv/bin/python -m tests.test_joint_smoke
```

This uses `openai/whisper-tiny` and `google/byt5-small` with synthetic audio. It does not replace end-to-end training on Sinhala data.

---

## Inference mental model (after Stage 2)

At inference time the joint model:

1. Runs **Whisper.generate** on Mel features.
2. Replays the token sequence through the Whisper decoder and takes the last hidden states.
3. **Projects** hidden states into the post-processor hidden size.
4. Runs **ByT5.generate** conditioned on those projected states as encoder outputs.

Training uses teacher-forced hypothesis tokens from Stage 0 so gradients do not pass through the discrete beam-search path used at inference.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Stage 0 Hub push denied | `HF_TOKEN` scope and repo id in `pseudo_dataset_name`. |
| Stage 1 “Neither parallel nor pseudo available” | `parallel_text_dataset` reachable and/or `pseudo_dataset_name` exists; token and dataset id correct. |
| CUDA OOM in Stage 2 | Lower `per_device_train_batch_size` in [src/config/training.py](src/config/training.py); set `freeze_whisper_encoder=True`; consider `apply_lora_to_whisper=False` only if you accept freezing Whisper updates. |
| Stage 2 loads wrong ByT5 | `postprocessor.warmstart_path` vs `hub_warmstart_repo`; ensure Stage 1 finished and the Hub revision is visible. |
| WER looks odd in eval | ByT5 byte sequences can be long; check `max_target_length` in [src/config/postprocessor.py](src/config/postprocessor.py) and `generation_max_length` in training config. |

---

## File map (quick reference)

| Area | Main files |
|------|------------|
| Task switch | [src/config/config.py](src/config/config.py) `RuntimeConfig.task` |
| ASR datasets | [src/config/dataset.py](src/config/dataset.py) |
| Whisper base / max tokens | [src/config/model.py](src/config/model.py) |
| LoRA | [src/config/lora.py](src/config/lora.py) |
| Trainer / Hub (shared) | [src/config/training.py](src/config/training.py) |
| Post-processor | [src/config/postprocessor.py](src/config/postprocessor.py) |
| Joint loss / Whisper checkpoint / joint Hub | [src/config/pipeline.py](src/config/pipeline.py) |
| Entrypoint | [main.py](main.py) |
| Joint facade | [src/joint_pipeline.py](src/joint_pipeline.py) |
| Stage 0 script | [src/stages/collect_pseudo_data.py](src/stages/collect_pseudo_data.py) |
| Stage 1 script | [src/stages/pretrain_postproc.py](src/stages/pretrain_postproc.py) |

---

## Optional: resume training

Set `CONFIG.runtime.resume_from_checkpoint` in [src/config/config.py](src/config/config.py) to a checkpoint path (string) or `True` to pick up the latest checkpoint in `output_dir`. This applies to the ASR trainer and the joint trainer when supported by the saved optimiser state.
