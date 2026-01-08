"""
Whisper Fine-Tuning Script for Sinhala Language
Demonstrates the Facade pattern for clean, readable code.
"""
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

import dotenv
logger.info("Loading environment variables...")
dotenv.load_dotenv()
logger.info("Importing dependencies...")

from src.huggingface import HuggingFaceAuthenticator
from src.data_loader import WhisperDataLoader
from src.data_preprocessor import DataPreprocessor
from src.asr_pipeline import WhisperASRPipeline
from src.config.config import CONFIG

logger.info("All dependencies loaded successfully")


def main():
    """Main execution function - highly readable thanks to facade pattern."""
    logger.info("=" * 60)
    logger.info("Whisper Fine-Tuning for Sinhala Language")
    logger.info("=" * 60)
    
    # Step 1: Authenticate with HuggingFace Hub
    logger.info("Step 1: Authenticating with HuggingFace Hub...")
    token = HuggingFaceAuthenticator.get_token_from_env()
    authenticator = HuggingFaceAuthenticator(token=token)
    
    if not authenticator.authenticate():
        logger.error("Authentication failed. Exiting.")
        return
    
    # Step 2: Load dataset
    logger.info("Step 2: Loading Sinhala ASR dataset...")
    data_loader = WhisperDataLoader(
        dataset_name=CONFIG.dataset.dataset_name,
        token=CONFIG.dataset.use_auth_token
    )

    logger.info(
        f"split train: {CONFIG.dataset.train_split}"
        f"split test: {CONFIG.dataset.test_split}"
    )
    dataset = data_loader.load_datasets(
        train_split=CONFIG.dataset.train_split,
        test_split=CONFIG.dataset.test_split
    )

    logger.info(f"Dataset loaded: {dataset}")
    
    # Step 3: Preprocess dataset (remove unnecessary columns)
    logger.info("Step 3: Preprocessing dataset...")
    preprocessor = DataPreprocessor(dataset=dataset)
    preprocessor.set_sample_rate("audio", CONFIG.dataset.sample_rate)
    dataset = preprocessor.get_dataset()
    logger.info("Dataset preprocessing complete")
    
    # Step 4: Initialize ASR pipeline
    logger.info("Step 4: Initializing ASR pipeline...")
    pipeline = WhisperASRPipeline(
        model_name=CONFIG.model.model_name,
        language=CONFIG.model.language,
        task=CONFIG.model.task
    )
    pipeline.initialize(dataset)
    logger.info("Pipeline initialization complete")
    
    # Step 5: Prepare data for training
    logger.info("Step 5: Preparing dataset for training...")
    prepared_dataset = pipeline.prepare_data(dataset)
    logger.info("Dataset preparation complete")
    
    # Step 6: Fine-tune the model
    logger.info("Step 6: Starting fine-tuning...")
    results = pipeline.finetune(prepared_dataset)
    
    # Step 7: Report results
    logger.info("=" * 60)
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Final WER: {results.get('eval_wer', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
