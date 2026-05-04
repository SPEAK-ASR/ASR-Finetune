"""Joint ASR + Post-Processor pipeline facade (Stage 2).

Mirrors the facade pattern of ``src/asr_pipeline.py`` but builds the joint model
(Whisper + Projection + Post-Processor) and trains it with the dual-tokenizer
collator and WER-on-post-output evaluator.
"""

from typing import Optional

import torch
from datasets import DatasetDict

from src.components.feature_extractor import FeatureExtractorComponent
from src.components.tokenizer import TokenizerComponent
from src.components.processor import ProcessorComponent
from src.components.model import ModelComponent
from src.components.postprocessor import PostProcessorComponent
from src.components.joint_model import JointASRPostProcessorModel
from src.components.joint_collator import JointDataCollator
from src.components.joint_evaluator import JointEvaluator
from src.components.joint_trainer import JointTrainer
from src.components.refresh_hyp_callback import RefreshHypothesesCallback
from src.data_preprocessor import DataPreprocessor
from src.config.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JointPipeline:
    """Facade for the joint ASR + Post-Processor training workflow."""

    def __init__(
        self,
        whisper_model_name: Optional[str] = None,
        postproc_warmstart: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
    ):
        self.whisper_model_name = (
            whisper_model_name or CONFIG.pipeline.whisper_warmstart_repo
        )
        self.postproc_warmstart = (
            postproc_warmstart or CONFIG.postprocessor.warmstart_path
        )
        self.language = language or CONFIG.model.language
        self.task = task or CONFIG.model.task

        self._feature_extractor = FeatureExtractorComponent(self.whisper_model_name)
        self._whisper_tokenizer = TokenizerComponent(
            self.whisper_model_name, self.language, self.task
        )
        self._whisper_processor = ProcessorComponent(
            self.whisper_model_name, self.language, self.task
        )
        self._whisper_model = ModelComponent(
            self.whisper_model_name, self.language, self.task
        )
        self._postprocessor = PostProcessorComponent(
            warmstart_path=self.postproc_warmstart
        )
        self._preprocessor = DataPreprocessor()

        self._joint_model: Optional[JointASRPostProcessorModel] = None
        self._initialized = False

        logger.info(
            f"JointPipeline created - Whisper: {self.whisper_model_name}, "
            f"Post-proc warmstart: {self.postproc_warmstart}"
        )

    def initialize(self) -> "JointPipeline":
        if self._initialized:
            return self

        logger.info("Initializing joint pipeline components...")
        self._feature_extractor.load()
        self._whisper_tokenizer.load()
        self._whisper_processor.create()
        whisper = self._whisper_model.load()
        postproc_model, postproc_tokenizer = self._postprocessor.load()

        self._joint_model = JointASRPostProcessorModel(
            whisper=whisper,
            postprocessor=postproc_model,
            whisper_tokenizer=self._whisper_tokenizer.get(),
            postproc_tokenizer=postproc_tokenizer,
        )

        self._initialized = True
        logger.info("Joint pipeline initialized")
        return self

    def prepare_data(self, dataset: DatasetDict) -> DatasetDict:
        """Run dual tokenisation / feature extraction on a Stage-0 pseudo dataset."""
        self._ensure_initialized()
        return self._preprocessor.prepare_joint_dataset(
            dataset=dataset,
            feature_extractor_component=self._feature_extractor,
            whisper_tokenizer_component=self._whisper_tokenizer,
            byt5_tokenizer=self._postprocessor.get_tokenizer(),
        )

    def finetune(self, dataset: DatasetDict) -> dict:
        """Train the joint model on a prepared dataset and return the training output."""
        self._ensure_initialized()

        whisper_tok = self._whisper_tokenizer.get()
        postproc_tok = self._postprocessor.get_tokenizer()

        collator = JointDataCollator(
            feature_extractor=self._feature_extractor.get(),
            whisper_tokenizer=whisper_tok,
            postproc_tokenizer=postproc_tok,
            decoder_start_token_id=self._joint_model.whisper.config.decoder_start_token_id,
        )

        evaluator = JointEvaluator(postproc_tokenizer=postproc_tok)

        extra_callbacks = []
        if CONFIG.pipeline.refresh_hyp_every_n_epochs > 0:
            extra_callbacks.append(
                RefreshHypothesesCallback(
                    train_dataset=dataset["train"],
                    whisper_tokenizer=whisper_tok,
                )
            )

        wrapper = JointTrainer(
            joint_model=self._joint_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            compute_metrics=evaluator.compute_metrics,
            postproc_tokenizer=postproc_tok,
            extra_callbacks=extra_callbacks,
        )

        logger.info("Starting joint fine-tuning...")
        results = wrapper.trainer.train(
            resume_from_checkpoint=CONFIG.runtime.resume_from_checkpoint
        )
        logger.info("Joint fine-tuning complete")

        if CONFIG.training.push_to_hub:
            logger.info(f"Pushing joint model to Hub: {CONFIG.pipeline.hub_joint_repo}")
            wrapper.trainer.push_to_hub()

        return results

    def _ensure_initialized(self):
        if not self._initialized:
            self.initialize()
