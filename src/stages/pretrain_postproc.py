"""Stage 1: pretrain the post-processor (ByT5) on {parallel_text_dataset} +
{Stage-0 pseudo pairs}.

Output: a fine-tuned ByT5 (or equivalent) that already understands Sinhala
text-polishing and has been exposed to this ASR's actual error distribution.
Used as warm-start for the joint model's post-processor in Stage 2.
"""

from dataclasses import replace
from typing import Optional

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from src.config.config import CONFIG
from src.components.postprocessor import PostProcessorComponent
from src.data_loader import WhisperDataLoader
from src.data_preprocessor import DataPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _build_text_pair_dataset(
    parallel: Optional[DatasetDict],
    pseudo: Optional[DatasetDict],
) -> DatasetDict:
    """Merge the two corpora into a unified DatasetDict with columns
    ``noisy_text`` / ``clean_text`` (standardised).
    """

    def _standardise_parallel(ds: DatasetDict) -> DatasetDict:
        noisy_col = CONFIG.dataset.parallel_noisy_column
        clean_col = CONFIG.dataset.parallel_clean_column
        return ds.rename_columns({noisy_col: "noisy_text", clean_col: "clean_text"})

    def _standardise_pseudo(ds: DatasetDict) -> DatasetDict:
        hyp_col = CONFIG.dataset.pseudo_hyp_column
        clean_col = CONFIG.dataset.pseudo_clean_column
        ds = ds.rename_columns({hyp_col: "noisy_text", clean_col: "clean_text"})
        drop = [c for c in ds["train"].column_names if c not in ("noisy_text", "clean_text")]
        if drop:
            ds = ds.remove_columns(drop)
        return ds

    splits: dict[str, list[Dataset]] = {"train": [], "test": []}

    if parallel is not None:
        p = _standardise_parallel(parallel)
        for s in splits:
            if s in p:
                splits[s].append(p[s])

    if pseudo is not None:
        ps = _standardise_pseudo(pseudo)
        for s in splits:
            if s in ps:
                splits[s].append(ps[s])

    out: dict[str, Dataset] = {}
    for s, parts in splits.items():
        if not parts:
            continue
        out[s] = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    # If there's no explicit test split, carve one out (5%).
    if "test" not in out and "train" in out:
        split_ds = out["train"].train_test_split(test_size=0.05, seed=42)
        out = {"train": split_ds["train"], "test": split_ds["test"]}

    return DatasetDict(out)


def pretrain_postprocessor(
    push_to_hub_repo: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Run Stage 1 and return the path (or Hub id) where the warm-started
    post-processor checkpoint was saved.
    """
    logger.info("=== Stage 1: Post-processor pretraining ===")

    loader = WhisperDataLoader()

    parallel = None
    try:
        parallel = loader.load_parallel_text_dataset()
    except Exception as e:
        logger.warning(f"Parallel text dataset unavailable: {e}")

    pseudo = None
    try:
        pseudo = loader.load_pseudo_dataset()
    except Exception as e:
        logger.warning(f"Pseudo dataset unavailable: {e}")

    if parallel is None and pseudo is None:
        raise RuntimeError(
            "Neither parallel_text_dataset nor pseudo_dataset is available - "
            "cannot pretrain post-processor."
        )

    merged = _build_text_pair_dataset(parallel, pseudo)
    logger.info(f"Merged text-pair corpus: { {k: len(v) for k, v in merged.items()} }")

    pp = PostProcessorComponent(model_name=CONFIG.postprocessor.model_name)
    model, tokenizer = pp.load()

    preprocessor = DataPreprocessor()
    prepared = preprocessor.prepare_parallel_text_dataset(
        dataset=merged,
        byt5_tokenizer=tokenizer,
        noisy_column="noisy_text",
        clean_column="clean_text",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest", return_tensors="pt"
    )

    repo = push_to_hub_repo or CONFIG.postprocessor.hub_warmstart_repo

    # Build Seq2Seq training args. We borrow safe defaults; for ByT5 we shrink
    # batch and generation length since byte-level sequences are long.
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{CONFIG.training.output_dir}/postproc_pretrain",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=3,
        bf16=CONFIG.training.bf16,
        predict_with_generate=False,
        push_to_hub=push_to_hub_repo is not None,
        hub_model_id=repo,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"] if any(
            True for _ in ()  # placeholder; rely on WANDB_DISABLED env if needed
        ) else "none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared.get("train"),
        eval_dataset=prepared.get("test"),
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(f"{training_args.output_dir}/final")

    if push_to_hub_repo is not None:
        trainer.push_to_hub()

    logger.info("=== Stage 1 complete ===")
    return f"{training_args.output_dir}/final"
