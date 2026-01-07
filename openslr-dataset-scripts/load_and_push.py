#!/usr/bin/env python3
"""
Load dataset from local folder, process it, and push to HuggingFace
"""
from datasets import load_dataset, Audio
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
DATASET_DIR = Path(__file__).parent / os.getenv('OUTPUT_DIR', 'sinhala-whisper')
REPO_ID = os.getenv('HF_REPO_ID')
TOKEN = os.getenv('HF_TOKEN')
PRIVATE = os.getenv('HF_PRIVATE', 'False').lower() == 'true'

# Validate token
if not TOKEN or TOKEN == 'your_huggingface_token_here':
    raise ValueError("Please set HF_TOKEN in .env file with a valid Hugging Face token")

def main():
    print("=" * 60)
    print("Loading and Processing Dataset")
    print("=" * 60)
    
    # Step 1: Load dataset from local CSV files
    print("\nLoading dataset from local folder...")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(DATASET_DIR / "data" / "train.csv"),
            "validation": str(DATASET_DIR / "data" / "validation.csv"),
            "test": str(DATASET_DIR / "data" / "test.csv")        }
    )
    
    print(f"Loaded dataset:")
    print(f"  - Train: {len(dataset['train'])} samples")
    print(f"  - Validation: {len(dataset['validation'])} samples")
    print(f"  - Test: {len(dataset['test'])} samples")
    print(f"\nColumns: {dataset['train'].column_names}")
    
    # Step 2: Update audio paths to absolute paths
    print("\nUpdating audio file paths...")
    
    def add_base_path(example):
        # Convert relative path to absolute path
        audio_path = DATASET_DIR / example["audio"].lstrip("../")
        example["audio"] = str(audio_path)
        return example
    
    dataset = dataset.map(add_base_path)

    # Step 3: Cast audio column to Audio feature with 16kHz sampling rate
    print("\nCasting audio column to Audio(sampling_rate=16000)...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("Audio column cast complete")
    
    # Verify the dataset structure
    print("\nFinal dataset structure:")
    print(f"  Columns: {dataset['train'].column_names}")
    print(f"  Features: {dataset['train'].features}")
    
    # Step 4: Push to HuggingFace
    print("\n" + "=" * 60)
    print("Pushing to HuggingFace Hub")
    print("=" * 60)
    print(f"\nRepository: {REPO_ID}")
    print("This will upload the dataset with audio files...")

    try:
        dataset.push_to_hub(
            REPO_ID,
            token=TOKEN,
            private=PRIVATE,
        )
        
        print("\n" + "=" * 60)
        print("Upload Complete!")
        print("=" * 60)
        print(f"\nYour dataset is now available at:")
        print(f"   https://huggingface.co/datasets/{REPO_ID}")
        
    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your token has WRITE permissions")
        print("3. Ensure you have access to the organization")

if __name__ == "__main__":
    main()
