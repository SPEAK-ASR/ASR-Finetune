#!/usr/bin/env python3
"""
Load dataset from local folder, process it, and push to HuggingFace
"""
from datasets import load_dataset, DatasetDict, Audio
from pathlib import Path
import os

# Configuration from environment variables
DATASET_DIR = Path(os.getenv("DATASET_DIR"))
REPO_ID = os.getenv("HF_REPO_ID")
TOKEN = os.getenv("HF_TOKEN")

def main():
    print("=" * 60)
    print("Loading and Processing Dataset")
    print("=" * 60)
    
    # Step 1: Load dataset from local CSV files
    print("Loading dataset from local folder...")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(DATASET_DIR / "data" / "train.csv"),
            "validation": str(DATASET_DIR / "data" / "validation.csv"),
            "test": str(DATASET_DIR / "data" / "test.csv")
        }
    )
    
    print(f"Loaded dataset:")
    print(f"  - Train: {len(dataset['train'])} samples")
    print(f"  - Validation: {len(dataset['validation'])} samples")
    print(f"  - Test: {len(dataset['test'])} samples")
    print(f"\nOriginal columns: {dataset['train'].column_names}")
    
    # Step 2: Rename columns
    print("Renaming columns...")
    print("  - 'file_name' → 'audio'")
    print("  - 'transcription' → 'text'")
    
    dataset = dataset.rename_column("file_name", "audio")
    dataset = dataset.rename_column("transcription", "text")
    
    print(f"New columns: {dataset['train'].column_names}")
    
    # Step 3: Update audio paths to absolute paths
    print("Updating audio file paths...")
    
    def add_base_path(example):
        # Convert relative path to absolute path
        audio_path = DATASET_DIR / example["audio"]
        example["audio"] = str(audio_path)
        return example
    
    dataset = dataset.map(add_base_path)
    print("Audio paths updated to absolute paths")
    
    # Step 4: Cast audio column to Audio feature with 16kHz sampling rate
    print("Casting audio column to Audio(sampling_rate=16000)...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("Audio column cast complete")
    
    # Verify the dataset structure
    print("Final dataset structure:")
    print(f"  Columns: {dataset['train'].column_names}")
    print(f"  Features: {dataset['train'].features}")
    
    # Step 5: Push to HuggingFace
    print("\n" + "=" * 60)
    print("Pushing to HuggingFace Hub")
    print("=" * 60)
    print(f"\nRepository: {REPO_ID}")
    print("This will upload the dataset with audio files...")
    
    try:
        dataset.push_to_hub(
            REPO_ID,
            token=TOKEN,
            private=False,
        )
        
        print("\n" + "=" * 60)
        print("Upload Complete!")
        print("=" * 60)
        print(f"\nYour dataset is now available at:")
        print(f"   https://huggingface.co/datasets/{REPO_ID}")
        print(f"\nLoad your dataset with:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{REPO_ID}')")
        print(f"\nThe audio files are automatically loaded at 16kHz!")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your token has WRITE permissions")
        print("3. Ensure you have access to the organization")

if __name__ == "__main__":
    main()
