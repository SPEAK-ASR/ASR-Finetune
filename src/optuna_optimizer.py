"""
Optuna Hyperparameter Optimization for LoRA Fine-tuning.

Uses Optuna to search for optimal LoRA parameters (rank, alpha, dropout)
by running short training trials and evaluating WER.

The search space is defined declaratively in SEARCH_SPACE — add or remove
entries there to control which hyperparameters are optimized.
"""

import gc
import json
import math
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import torch
from datasets import DatasetDict
from optuna.storages import JournalFileStorage, JournalStorage
from transformers import TrainerCallback

from src.asr_pipeline import WhisperASRPipeline
from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Path for the shared Optuna journal file (used for multi-process coordination)
OPTUNA_JOURNAL_PATH = os.path.join(CONFIG.paths.log_dir, "optuna_journal.log")


# ---------------------------------------------------------------------------
# Declarative search space
# ---------------------------------------------------------------------------
# Each entry maps to a CONFIG sub-object (target) and an attribute (name).
# Supported types: "int", "float", "categorical"
#
# To add a new parameter (e.g. learning_rate), simply append a dict:
#   {"name": "learning_rate", "target": "training", "type": "float",
#    "low": 1e-6, "high": 1e-3, "log": True},
#
# To remove a parameter, delete or comment out its entry.
# ---------------------------------------------------------------------------
SEARCH_SPACE: list[dict[str, Any]] = [
    {
        "name": "r",
        "target": "lora",
        "type": "int",
        "low": 4,
        "high": 128,
        "log": True,
    },
    {
        "name": "lora_alpha",
        "target": "lora",
        "type": "int",
        "low": 8,
        "high": 256,
        "log": True,
    },
    {
        "name": "lora_dropout",
        "target": "lora",
        "type": "float",
        "low": 0.0,
        "high": 0.3,
        "log": False,
    },
    # Uncomment to include learning_rate in the search:
    # {
    #     "name": "learning_rate",
    #     "target": "training",
    #     "type": "float",
    #     "low": 1e-6,
    #     "high": 1e-3,
    #     "log": True,
    # },
]


class OptunaCallback(TrainerCallback):
    """Custom callback to integrate Optuna with Transformers Trainer.
    
    Reports metrics to Optuna and enables pruning of unpromising trials.
    """

    def __init__(self, trial: optuna.Trial, metric: str = "eval_wer"):
        self.trial = trial
        self.metric = metric

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Report metric to Optuna after each evaluation."""
        if self.metric in metrics:
            # Report the metric value at the current step
            self.trial.report(metrics[self.metric], state.global_step)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def _suggest_params(
    trial: optuna.Trial,
    search_space: list[dict[str, Any]],
) -> dict[str, Any]:
    """Suggest hyperparameters from *search_space* and apply them to CONFIG.

    Returns a dict of ``{name: suggested_value}`` for logging purposes.
    """
    suggested: dict[str, Any] = {}

    for entry in search_space:
        name = entry["name"]
        target_cfg = getattr(CONFIG, entry["target"])
        log = entry.get("log", False)

        if entry["type"] == "int":
            value = trial.suggest_int(name, entry["low"], entry["high"], log=log)
        elif entry["type"] == "float":
            value = trial.suggest_float(name, entry["low"], entry["high"], log=log)
        elif entry["type"] == "categorical":
            value = trial.suggest_categorical(name, entry["choices"])
        else:
            raise ValueError(f"Unsupported search‑space type: {entry['type']}")

        setattr(target_cfg, name, value)
        suggested[name] = value

    return suggested


class OptunaOptimizer:
    """Run an Optuna study to find the best LoRA hyperparameters."""

    def __init__(self, dataset: DatasetDict) -> None:
        self.dataset = dataset
        self.trial_epochs: float = CONFIG.runtime.optuna_trial_epochs

        # When launched via run_optuna_parallel.sh, each worker receives
        # a per-worker trial count via OPTUNA_TRIALS_PER_WORKER env var.
        # Otherwise, fall back to the configured total.
        per_worker = os.environ.get("OPTUNA_TRIALS_PER_WORKER")
        self.n_trials: int = (
            int(per_worker) if per_worker else CONFIG.runtime.optuna_n_trials
        )

        # GPU info for logging
        self._gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        if torch.cuda.is_available():
            logger.info(
                f"GPU worker — CUDA_VISIBLE_DEVICES={self._gpu_id}, "
                f"device={torch.cuda.get_device_name(0)}"
            )

        # Snapshot the original values so we can restore them after the study.
        self._original_values: dict[str, dict[str, Any]] = {}
        for entry in SEARCH_SPACE:
            target = entry["target"]
            name = entry["name"]
            self._original_values.setdefault(target, {})[name] = deepcopy(
                getattr(getattr(CONFIG, target), name)
            )

        # Also snapshot training overrides that we change per‑trial.
        self._original_training = {
            "num_train_epochs": deepcopy(CONFIG.training.num_train_epochs),
            "push_to_hub": deepcopy(CONFIG.training.push_to_hub),
            "report_to": deepcopy(CONFIG.training.report_to),
            "output_dir": deepcopy(CONFIG.training.output_dir),
        }

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def objective(self, trial: optuna.Trial) -> float:
        """Train one trial and return the eval WER (lower is better)."""

        # 1) Suggest & apply search‑space params
        suggested = _suggest_params(trial, SEARCH_SPACE)
        logger.info(f"Trial {trial.number} — params: {suggested}")

        # 2) Override training config for a short, silent trial
        CONFIG.training.num_train_epochs = self.trial_epochs
        CONFIG.training.push_to_hub = False
        CONFIG.training.report_to = []
        CONFIG.training.output_dir = f"checkpoints/optuna_trial_{trial.number}"

        try:
            # 3) Fresh pipeline (forces fresh model load)
            pipeline = WhisperASRPipeline(
                model_name=CONFIG.model.base_model_name,
                language=CONFIG.model.language,
                task=CONFIG.model.task,
            )
            pipeline.initialize()

            # 4) Setup data collator & create trainer
            pipeline._setup_data_collator()
            trainer = pipeline._create_trainer(
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
            )

            # 5) Inject Optuna pruning callback
            trainer.add_callback(OptunaCallback(trial))

            # 6) Train
            logger.info(f"Trial {trial.number} — starting training "
                        f"({self.trial_epochs} epoch(s))…")
            trainer.train()

            # 7) Evaluate
            metrics = trainer.evaluate()
            wer = metrics.get("eval_wer", float("inf"))
            logger.info(f"Trial {trial.number} — eval WER: {wer:.4f}")

            return wer

        except optuna.TrialPruned:
            raise  # let Optuna handle it

        except Exception:
            logger.exception(f"Trial {trial.number} failed")
            raise

        finally:
            # 8) Cleanup GPU memory & trial checkpoints
            self._cleanup_trial(trial.number)

    # ------------------------------------------------------------------
    # Run the full study
    # ------------------------------------------------------------------
    def run(self) -> optuna.Study:
        """Create an Optuna study, optimize, save results, and return it."""

        logger.info(f"{'=' * 60}")
        logger.info("Starting Optuna hyperparameter optimization")
        logger.info(f"Trials (this worker): {self.n_trials} | "
                    f"Epochs/trial: {self.trial_epochs} | GPU: {self._gpu_id}")
        logger.info(f"Search space: {[e['name'] for e in SEARCH_SPACE]}")
        logger.info(f"{'=' * 60}")

        # Use JournalFileStorage so multiple GPU workers can share one study.
        # Works fine for single-process too (just a local file).
        storage = JournalStorage(
            JournalFileStorage(OPTUNA_JOURNAL_PATH),
        )

        study = optuna.create_study(
            direction="minimize",
            study_name="lora_optimization",
            storage=storage,
            load_if_exists=True,  # workers join the same study
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
            ),
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            gc_after_trial=True,
        )

        # Log results
        logger.info(f"{'=' * 60}")
        logger.info("Optuna optimization complete!")
        logger.info(f"Best trial: #{study.best_trial.number}")
        logger.info(f"Best WER:   {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"{'=' * 60}")

        self._save_results(study)
        self._restore_config()

        return study

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cleanup_trial(self, trial_number: int) -> None:
        """Free GPU memory and optionally remove trial checkpoint dir."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ckpt_dir = Path(f"checkpoints/optuna_trial_{trial_number}")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.debug(f"Removed checkpoint dir: {ckpt_dir}")

    def _save_results(self, study: optuna.Study) -> None:
        """Persist study results to ``logs/optuna_results.json``."""
        log_dir = Path(CONFIG.paths.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "best_trial_number": study.best_trial.number,
            "best_wer": study.best_value,
            "best_params": study.best_params,
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }

        out_path = log_dir / "optuna_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")

        # Optuna visualization (optional — requires plotly)
        self._save_visualizations(study, log_dir)

    @staticmethod
    def _save_visualizations(study: optuna.Study, log_dir: Path) -> None:
        """Generate Optuna HTML plots if plotly is available."""
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_contour,
                plot_slice,
            )

            plots = {
                "optimization_history": plot_optimization_history,
                "param_importances": plot_param_importances,
                "contour": plot_contour,
                "slice": plot_slice,
            }

            for name, plot_fn in plots.items():
                try:
                    fig = plot_fn(study)
                    path = log_dir / f"optuna_{name}.html"
                    fig.write_html(str(path))
                    logger.info(f"Saved plot: {path}")
                except Exception:
                    logger.debug(f"Could not generate plot '{name}'", exc_info=True)

        except ImportError:
            logger.warning(
                "plotly not installed — skipping Optuna visualization. "
                "Install with: pip install plotly"
            )

    def _restore_config(self) -> None:
        """Restore CONFIG values that were modified during optimization."""
        # Restore search‑space params
        for target, attrs in self._original_values.items():
            cfg = getattr(CONFIG, target)
            for name, value in attrs.items():
                setattr(cfg, name, value)

        # Restore training overrides
        for name, value in self._original_training.items():
            setattr(CONFIG.training, name, value)

        logger.info("Original CONFIG values restored")
