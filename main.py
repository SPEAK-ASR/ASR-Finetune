"""
Whisper Fine-Tuning Script for Sinhala Language
Demonstrates the Facade pattern for clean, readable code.
"""
import os

# Check if this is the main process (rank 0) for logging
IS_MAIN_PROCESS = int(os.environ.get("LOCAL_RANK", 0)) == 0

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

import dotenv
if IS_MAIN_PROCESS:
    logger.info("Loading environment variables...")
dotenv.load_dotenv()

from src.config.config import CONFIG
if IS_MAIN_PROCESS:
    logger.info(f"Cache configured: datasets={CONFIG.paths.cache_dir}, models={CONFIG.paths.model_cache_dir}")

from datasets import DatasetDict
from src.huggingface import HuggingFaceAuthenticator
from src.data_loader import WhisperDataLoader
from src.data_preprocessor import DataPreprocessor
from src.asr_pipeline import WhisperASRPipeline
from src.joint_pipeline import JointPipeline
from src.stages.collect_pseudo_data import collect_pseudo_data
from src.stages.pretrain_postproc import pretrain_postprocessor
from src.config.wandb_config import WandbAuthenticator

if IS_MAIN_PROCESS:
    logger.info("All dependencies loaded successfully")

def _create_prepared_dataset(token: str, dataset: DatasetDict) -> None:
    if IS_MAIN_PROCESS:
        logger.info("Starting dataset preparation process...")
    
    if IS_MAIN_PROCESS:
        logger.info("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(dataset=dataset)
    preprocessor.set_sample_rate("audio", CONFIG.dataset.sample_rate)
    if IS_MAIN_PROCESS:
        logger.info(f"Sample rate set to: {CONFIG.dataset.sample_rate} Hz")
    dataset = preprocessor.get_dataset()
    if IS_MAIN_PROCESS:
        logger.info("Preprocessing completed")

    if IS_MAIN_PROCESS:
        logger.info(f"Initializing Whisper ASR Pipeline (Model: {CONFIG.model.base_model_name})...")
    pipeline = WhisperASRPipeline(
        model_name=CONFIG.model.base_model_name,
        language=CONFIG.model.language,
        task=CONFIG.model.task
    )
    pipeline.initialize(dataset)
    if IS_MAIN_PROCESS:
        logger.info("Pipeline initialized successfully")

    if IS_MAIN_PROCESS:
        logger.info("Preparing dataset for fine-tuning...")
    prepared_dataset = pipeline.prepare_data(dataset)
    if IS_MAIN_PROCESS:
        logger.info("Dataset preparation completed")

    repo_name = f"{CONFIG.dataset.dataset_name}-preprocessed"
    if IS_MAIN_PROCESS:
        logger.info(f"Pushing prepared dataset to Hub: {repo_name}")
    prepared_dataset.push_to_hub(
        repo_name,
        token=token,
        private=False,
    )
    if IS_MAIN_PROCESS:
        logger.info(f"Dataset successfully pushed to Hub: {repo_name}")

def _finetune_asr_model(token: str, dataset: DatasetDict) -> None:
    if IS_MAIN_PROCESS:
        logger.info("Starting model fine-tuning process...")

    if IS_MAIN_PROCESS:
        logger.info(f"Initializing Whisper ASR Pipeline (Model: {CONFIG.model.base_model_name})...")
    pipeline = WhisperASRPipeline(
        model_name=CONFIG.model.base_model_name,
        language=CONFIG.model.language,
        task=CONFIG.model.task
    )
    if IS_MAIN_PROCESS:
        logger.info(f"Language: {CONFIG.model.language} | Task: {CONFIG.model.task}")
    pipeline.initialize()
    if IS_MAIN_PROCESS:
        logger.info("Pipeline initialized successfully")

    if IS_MAIN_PROCESS:
        logger.info("Starting fine-tuning...")
    results = pipeline.finetune(dataset)
    if IS_MAIN_PROCESS:
        logger.info("Fine-tuning completed")
    
    if IS_MAIN_PROCESS:
        logger.info(f"{'=' * 40}")
        logger.info(f"Training Results:")
        logger.info(f"Final results: {results}")
        logger.info(f"{'=' * 40}")


def _collect_pseudo_data(token: str, dataset: DatasetDict) -> None:
    """Stage 0 task: run fine-tuned Whisper over the ASR dataset and push the
    resulting (audio, asr_hyp_text, clean_text) corpus to the Hub."""
    if IS_MAIN_PROCESS:
        logger.info("Starting Stage 0: pseudo-data collection...")
    collect_pseudo_data(
        dataset=dataset,
        push_to_hub_repo=CONFIG.dataset.pseudo_dataset_name,
        token=token,
    )


def _pretrain_postproc(token: str) -> None:
    """Stage 1 task: pretrain the post-processor on parallel + pseudo pairs."""
    if IS_MAIN_PROCESS:
        logger.info("Starting Stage 1: post-processor pretraining...")
    pretrain_postprocessor(
        push_to_hub_repo=CONFIG.postprocessor.hub_warmstart_repo,
        token=token,
    )


def _finetune_joint_pipeline(token: str) -> None:
    """Stage 2 task: joint fine-tuning of Whisper + post-processor via hidden-state
    coupling. Uses the Stage-0 pseudo dataset (which has audio + ASR hyps + gold)."""
    if IS_MAIN_PROCESS:
        logger.info("Starting Stage 2: joint pipeline fine-tuning...")

    loader = WhisperDataLoader()
    pseudo_dataset = loader.load_pseudo_dataset()

    pipeline = JointPipeline(
        whisper_model_name=CONFIG.pipeline.whisper_warmstart_repo,
        postproc_warmstart=CONFIG.postprocessor.warmstart_path
        or CONFIG.postprocessor.hub_warmstart_repo,
    )
    pipeline.initialize()

    prepared = pipeline.prepare_data(pseudo_dataset)

    # Stage-0 dataset may only have a "train" split if built from a monolithic
    # source. Carve out a held-out eval split.
    if "test" not in prepared:
        split = prepared["train"].train_test_split(test_size=0.02, seed=42)
        prepared = DatasetDict(train=split["train"], test=split["test"])

    results = pipeline.finetune(prepared)

    if IS_MAIN_PROCESS:
        logger.info(f"{'=' * 40}")
        logger.info(f"Joint Training Results: {results}")
        logger.info(f"{'=' * 40}")


def main():
    """Main execution function - highly readable thanks to facade pattern."""
    if IS_MAIN_PROCESS:
        logger.info(f"{'=' * 60}\nWhisper Fine-Tuning for Sinhala Language\n{'=' * 60}")

    # Authenticate with HuggingFace
    if IS_MAIN_PROCESS:
        logger.info("Authenticating with HuggingFace...")
    token = HuggingFaceAuthenticator.get_token_from_env()
    authenticator = HuggingFaceAuthenticator(token=token)
    
    if not authenticator.authenticate():
        if IS_MAIN_PROCESS:
            logger.error("Authentication failed. Exiting.")
        return
    if IS_MAIN_PROCESS:
        logger.info("Successfully authenticated with HuggingFace")

    # Authenticate with Weights & Biases
    if IS_MAIN_PROCESS:
        logger.info("Authenticating with Weights & Biases...")
    wandb_api_key = WandbAuthenticator.get_api_key_from_env()
    wandb_authenticator = WandbAuthenticator(api_key=wandb_api_key)
    if not wandb_authenticator.authenticate():
        if IS_MAIN_PROCESS:
            logger.error("W&B Authentication failed. Exiting.")
        return
    run = wandb_authenticator.init_run(project=f"whisper-finetune-sinhala", entity="SPEAK-ASR-uom")
    if IS_MAIN_PROCESS:
        logger.info("Successfully initialized W&B")
    
    # Tasks that need the raw ASR dataset loaded upfront.
    tasks_needing_asr_dataset = {
        "prepare_dataset",
        "finetune_asr_model",
        "collect_pseudo_data",
    }

    if IS_MAIN_PROCESS:
        logger.info(f"Task selected: {CONFIG.runtime.task}")

    dataset = None
    if CONFIG.runtime.task in tasks_needing_asr_dataset:
        dataset_names = [ds.dataset_name for ds in CONFIG.dataset.datasets]
        if IS_MAIN_PROCESS:
            logger.info(f"Loading {len(CONFIG.dataset.datasets)} dataset(s): {dataset_names}")
        data_loader = WhisperDataLoader()
        dataset = data_loader.load_datasets()

        if IS_MAIN_PROCESS:
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Train samples: {len(dataset['train'])}")
            logger.info(f"Test samples: {len(dataset['test'])}")
            logger.info(f"Dataset structure: {dataset}")

    if CONFIG.runtime.task == "prepare_dataset":
        _create_prepared_dataset(token, dataset)
        if IS_MAIN_PROCESS:
            logger.info("Dataset preparation task completed successfully")
    elif CONFIG.runtime.task == "finetune_asr_model":
        _finetune_asr_model(token, dataset)
        if IS_MAIN_PROCESS:
            logger.info("Model fine-tuning task completed successfully")
    elif CONFIG.runtime.task == "collect_pseudo_data":
        _collect_pseudo_data(token, dataset)
        if IS_MAIN_PROCESS:
            logger.info("Pseudo-data collection task completed successfully")
    elif CONFIG.runtime.task == "pretrain_postproc":
        _pretrain_postproc(token)
        if IS_MAIN_PROCESS:
            logger.info("Post-processor pretraining task completed successfully")
    elif CONFIG.runtime.task == "finetune_joint_pipeline":
        _finetune_joint_pipeline(token)
        if IS_MAIN_PROCESS:
            logger.info("Joint pipeline fine-tuning task completed successfully")
    else:
        if IS_MAIN_PROCESS:
            logger.error(f"Unknown task: {CONFIG.runtime.task}")
        return
    
    if IS_MAIN_PROCESS:
        logger.info(f"{'=' * 60}\nExecution completed successfully\n{'=' * 60}")


if __name__ == "__main__":
    main()
