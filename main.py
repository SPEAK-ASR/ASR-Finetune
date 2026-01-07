"""
Whisper Fine-Tuning Script for Sinhala Language
This script provides a clean, class-based implementation for fine-tuning
OpenAI's Whisper model on Sinhala language data.
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

logger.info("All dependencies loaded successfully")


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Whisper Fine-Tuning for Sinhala Language")
    logger.info("=" * 60)
    
    # Initialize HuggingFace authenticator
    logger.info("Step 1: Authenticating with HuggingFace Hub...")
    token = HuggingFaceAuthenticator.get_token_from_env()
    authenticator = HuggingFaceAuthenticator(token=token)
    
    if not authenticator.authenticate():
        logger.error("Unable to proceed without authentication. Exiting.")
        return
    
    # Load dataset
    logger.info("Step 2: Loading Sinhala ASR dataset...")
    data_loader = WhisperDataLoader(
        dataset_name="SPEAK-ASR/youtube-sinhala-asr",
        token=True
    )
    
    dataset = data_loader.load_datasets()
    print(dataset)
    
    logger.info("Dataset loaded successfully.")

    # Preprocess dataset
    logger.info("Step 3: Preprocessing dataset...")

    preprocessor = DataPreprocessor(dataset=dataset)
    columns_to_remove = ["speaker_gender"]
    processed_dataset = preprocessor.remove_columns(columns_to_remove)

    # Setup ASR pipeline
    logger.info("Step 4: Setting up ASR pipeline...")
    asr_pipeline = WhisperASRPipeline(
        model_name="openai/whisper-small",
        language="Sinhala",
        task="transcribe"
    )
    processor = asr_pipeline.setup_pipeline(dataset=processed_dataset)
    logger.info("ASR pipeline setup complete.")
    prepared_dataset = asr_pipeline.prepare_dataset(processed_dataset)
    logger.info(f"Dataset preparation complete.\nPrepared dataset: {prepared_dataset['train'][0]}")

if __name__ == "__main__":
    main()
