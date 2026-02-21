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
    
    # Load dataset(s)
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
    
    if IS_MAIN_PROCESS:
        logger.info(f"Task selected: {CONFIG.runtime.task}")
    if CONFIG.runtime.task == "prepare_dataset":
        _create_prepared_dataset(token, dataset)
        if IS_MAIN_PROCESS:
            logger.info("Dataset preparation task completed successfully")
    elif CONFIG.runtime.task == "finetune_asr_model":
        _finetune_asr_model(token, dataset)
        if IS_MAIN_PROCESS:
            logger.info("Model fine-tuning task completed successfully")
    else:
        if IS_MAIN_PROCESS:
            logger.error(f"Unknown task: {CONFIG.runtime.task}")
        return
    
    if IS_MAIN_PROCESS:
        logger.info(f"{'=' * 60}\nExecution completed successfully\n{'=' * 60}")


if __name__ == "__main__":
    main()
