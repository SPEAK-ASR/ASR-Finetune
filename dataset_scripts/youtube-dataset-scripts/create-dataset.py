import pandas as pd
import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from collections import Counter

# Configuration from environment variables
BASE_DIR = Path(__file__).parent
CSV_FILE = Path(os.getenv("CSV_FILE", BASE_DIR / "db_export_rows.csv"))
AUDIO_SOURCE_DIR = Path(os.getenv("AUDIO_SOURCE_DIR"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "sinhala-whisper"))

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

def create_directory_structure():
    """Create the required directory structure"""
    dirs = [
        OUTPUT_DIR / "data",
        OUTPUT_DIR / "audio" / "train",
        OUTPUT_DIR / "audio" / "validation",
        OUTPUT_DIR / "audio" / "test"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created")

def load_and_prepare_data():
    """Load CSV and prepare data"""
    print("\nLoading CSV data...")
    df = pd.read_csv(CSV_FILE)
    
    # Remove any rows with missing values
    df = df.dropna()
    
    print(f"Loaded {len(df)} records")
    print(f"  - Male speakers: {(df['speaker_gender'] == 'male').sum()}")
    print(f"  - Female speakers: {(df['speaker_gender'] == 'female').sum()}")
    
    return df

def split_dataset(df):
    """Split dataset into train, validation, and test sets"""
    print("\nSplitting dataset...")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_RATIO, 
        random_state=42,
        stratify=df['speaker_gender']  # Maintain gender distribution
    )
    
    # Second split: separate validation from train
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_RATIO/(TRAIN_RATIO + VAL_RATIO),
        random_state=42,
        stratify=train_val_df['speaker_gender']
    )
    
    print(f"Split completed:")
    print(f"  - Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def copy_audio_files(df, split_name):
    """Copy audio files to the appropriate directory"""
    target_dir = OUTPUT_DIR / "audio" / split_name
    
    print(f"\nCopying {split_name} audio files...")
    copied = 0
    missing = 0
    
    for filename in df['audio_filename']:
        source = AUDIO_SOURCE_DIR / filename
        target = target_dir / filename
        
        if source.exists():
            shutil.copy2(source, target)
            copied += 1
        else:
            missing += 1
            print(f"Warning: Missing file {filename}")
    
    print(f"✓ Copied {copied} files")
    if missing > 0:
        print(f"{missing} files were missing")
    
    return copied, missing

def create_csv_files(train_df, val_df, test_df):
    """Create CSV files for each split"""
    print("\nCreating CSV files...")
    
    # Reset index and add file_name column (relative path to audio)
    for df, split_name in [(train_df, "train"), (val_df, "validation"), (test_df, "test")]:
        df_copy = df.copy()
        df_copy = df_copy.reset_index(drop=True)
        
        # Create proper format for HuggingFace datasets
        # Columns: file_name, transcription, speaker_gender
        df_output = pd.DataFrame({
            'file_name': df_copy['audio_filename'].apply(lambda x: f"audio/{split_name}/{x}"),
            'transcription': df_copy['transcription'],
            'speaker_gender': df_copy['speaker_gender']
        })
        
        output_path = OUTPUT_DIR / "data" / f"{split_name}.csv"
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Created {split_name}.csv")

def main():
    print("=" * 60)
    print("Creating Sinhala Whisper Fine-tuning Dataset")
    print("=" * 60)
    
    # Check if source files exist
    if not CSV_FILE.exists():
        print(f"Error: CSV file not found at {CSV_FILE}")
        return
    
    if not AUDIO_SOURCE_DIR.exists():
        print(f"Error: Audio directory not found at {AUDIO_SOURCE_DIR}")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Copy audio files
    for df_split, split_name in [(train_df, "train"), (val_df, "validation"), (test_df, "test")]:
        copy_audio_files(df_split, split_name)
    
    # Create CSV files
    create_csv_files(train_df, val_df, test_df)
    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print(f"\nNext steps to upload to HuggingFace:")
    print(f"\nReview the dataset structure in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
