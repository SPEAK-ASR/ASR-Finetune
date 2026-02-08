"""
Whisper Fine-Tuning Script for Sinhala Language
Demonstrates the Facade pattern for clean, readable code.
"""
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

import dotenv
logger.info("Loading environment variables...")
dotenv.load_dotenv()

from src.config.config import CONFIG
logger.info(f"Cache configured: datasets={CONFIG.paths.cache_dir}, models={CONFIG.paths.model_cache_dir}")

from datasets import DatasetDict
from src.config.dataset import get_dataset_names
from src.huggingface import HuggingFaceAuthenticator
from src.data_loader import WhisperDataLoader
from src.data_preprocessor import DataPreprocessor
from src.asr_pipeline import WhisperASRPipeline
from src.config.wandb_config import WandbAuthenticator

logger.info("All dependencies loaded successfully")

def _create_prepared_dataset(token: str, dataset: DatasetDict) -> None:
    logger.info("Starting dataset preparation process...")
    
    logger.info("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(dataset=dataset)
    preprocessor.set_sample_rate("audio", CONFIG.dataset.sample_rate)
    logger.info(f"Sample rate set to: {CONFIG.dataset.sample_rate} Hz")
    dataset = preprocessor.get_dataset()
    logger.info("Preprocessing completed")

    logger.info(f"Initializing Whisper ASR Pipeline (Model: {CONFIG.model.base_model_name})...")
    pipeline = WhisperASRPipeline(
        model_name=CONFIG.model.base_model_name,
        language=CONFIG.model.language,
        task=CONFIG.model.task
    )
    pipeline.initialize(dataset)
    logger.info("Pipeline initialized successfully")

    logger.info("Preparing dataset for fine-tuning...")
    prepared_dataset = pipeline.prepare_data(dataset)
    logger.info("Dataset preparation completed")

    repo_name = f"{get_dataset_names()[0]}-preprocessed"
    logger.info(f"Pushing prepared dataset to Hub: {repo_name}")
    prepared_dataset.push_to_hub(
        repo_name,
        token=token,
        private=False,
    )
    logger.info(f"Dataset successfully pushed to Hub: {repo_name}")

def _finetune_asr_model(token: str, dataset: DatasetDict) -> None:
    logger.info("Starting model fine-tuning process...")

    logger.info(f"Initializing Whisper ASR Pipeline (Model: {CONFIG.model.base_model_name})...")
    pipeline = WhisperASRPipeline(
        model_name=CONFIG.model.base_model_name,
        language=CONFIG.model.language,
        task=CONFIG.model.task
    )
    logger.info(f"Language: {CONFIG.model.language} | Task: {CONFIG.model.task}")
    pipeline.initialize()
    logger.info("Pipeline initialized successfully")

    logger.info("Starting fine-tuning...")
    results = pipeline.finetune(dataset)
    logger.info("Fine-tuning completed")
    
    logger.info(f"{'=' * 40}")
    logger.info(f"Training Results:")
    logger.info(f"Final results: {results}")
    logger.info(f"{'=' * 40}")



def main():
    """Main execution function - highly readable thanks to facade pattern."""
    logger.info(f"{'=' * 60}\nWhisper Fine-Tuning for Sinhala Language\n{'=' * 60}")

    # Authenticate with HuggingFace
    logger.info("Authenticating with HuggingFace...")
    token = HuggingFaceAuthenticator.get_token_from_env()
    authenticator = HuggingFaceAuthenticator(token=token)
    
    if not authenticator.authenticate():
        logger.error("Authentication failed. Exiting.")
        return
    logger.info("Successfully authenticated with HuggingFace")

    # Authenticate with Weights & Biases
    logger.info("Authenticating with Weights & Biases...")
    wandb_api_key = WandbAuthenticator.get_api_key_from_env()
    wandb_authenticator = WandbAuthenticator(api_key=wandb_api_key)
    if not wandb_authenticator.authenticate():
        logger.error("W&B Authentication failed. Exiting.")
        return
    run = wandb_authenticator.init_run(project=f"whisper-finetune-sinhala", entity="SPEAK-ASR-uom")
    logger.info("Successfully initialized W&B")
    
    # Load dataset(s)
    dataset_names = [ds.dataset_name for ds in CONFIG.dataset.datasets]
    logger.info(f"Loading {len(CONFIG.dataset.datasets)} dataset(s): {dataset_names}")
    data_loader = WhisperDataLoader()

    dataset = data_loader.load_datasets()

    logger.info(f"Dataset loaded successfully")
    if CONFIG.dataset.streaming:
        logger.info("Dataset is in streaming mode - samples will be loaded on-the-fly during training")
    else:
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Test samples: {len(dataset['test'])}")
    logger.info(f"Dataset structure: {dataset}")
    
    logger.info(f"Task selected: {CONFIG.runtime.task}")
    if CONFIG.runtime.task == "prepare_dataset":
        _create_prepared_dataset(token, dataset)
        logger.info("Dataset preparation task completed successfully")
    elif CONFIG.runtime.task == "finetune_asr_model":
        _finetune_asr_model(token, dataset)
        logger.info("Model fine-tuning task completed successfully")
    else:
        logger.error(f"Unknown task: {CONFIG.runtime.task}")
        return
    
    logger.info(f"{'=' * 60}\nExecution completed successfully\n{'=' * 60}")


if __name__ == "__main__":
    main()
