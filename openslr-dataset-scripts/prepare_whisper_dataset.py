import os
import csv
import shutil
from pathlib import Path
import random
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Increase CSV field size limit
csv.field_size_limit(10485760)  # 10MB should be enough

# Configuration from environment variables
TRAIN_RATIO = float(os.getenv('TRAIN_RATIO', '0.7'))
VAL_RATIO = float(os.getenv('VAL_RATIO', '0.1'))
TEST_RATIO = float(os.getenv('TEST_RATIO', '0.2'))

# Paths
BASE_DIR = Path(__file__).parent
TSV_FILE = BASE_DIR / os.getenv('TSV_FILE', 'utt_spk_text.tsv')
SOURCE_AUDIO_DIR = BASE_DIR / os.getenv('SOURCE_AUDIO_DIR', 'data')
OUTPUT_DIR = BASE_DIR / os.getenv('OUTPUT_DIR', 'sinhala-whisper')

# Create output directory structure
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"

def create_directory_structure():
    """Create the required directory structure for the dataset"""
    print("Creating directory structure...")
    
    # Create main directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_AUDIO_DIR.mkdir(exist_ok=True)
    
    # Create split directories
    for split in ['train', 'validation', 'test']:
        (OUTPUT_AUDIO_DIR / split).mkdir(exist_ok=True)
    
    print("Directory structure created!")

def read_tsv_file():
    """Read the TSV file and return a list of (utterance_id, speaker_id, text)"""
    print("Reading TSV file...")
    data = []
    
    with open(TSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                utt_id, spk_id, text = row[0], row[1], row[2]
                data.append((utt_id, spk_id, text))
    
    print(f"Read {len(data)} utterances from TSV file")
    return data

def find_audio_file(utt_id):
    """Find the audio file for a given utterance ID"""
    # Audio files are organized in subdirectories based on first 2 characters
    prefix = utt_id[:2]
    audio_path = SOURCE_AUDIO_DIR / prefix / f"{utt_id}.flac"
    
    if audio_path.exists():
        return audio_path
    return None

def split_data(data):
    """Split data into train, validation, and test sets"""
    print("Splitting data into train/validation/test sets...")
    
    # Shuffle the data
    random.seed(42)  # For reproducibility
    random.shuffle(data)
    
    total_samples = len(data)
    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

def process_split(split_name, data_split):
    """Process a single split: copy audio files and create CSV"""
    print(f"\nProcessing {split_name} split...")
    
    csv_file = OUTPUT_DATA_DIR / f"{split_name}.csv"
    audio_output_dir = OUTPUT_AUDIO_DIR / split_name
    
    csv_data = []
    missing_files = 0
    
    for idx, (utt_id, spk_id, text) in enumerate(tqdm(data_split, desc=f"Processing {split_name}")):
        # Find source audio file
        source_audio = find_audio_file(utt_id)
        
        if source_audio is None:
            missing_files += 1
            continue
        
        # Create new filename with sequential numbering
        new_filename = f"{idx+1:06d}.flac"
        dest_audio = audio_output_dir / new_filename
        
        # Copy audio file
        try:
            shutil.copy2(source_audio, dest_audio)
            
            # Add to CSV data
            # Relative path from the data directory
            relative_audio_path = f"../audio/{split_name}/{new_filename}"
            csv_data.append({
                'audio': relative_audio_path,
                'text': text
            })
        except Exception as e:
            print(f"Error copying {source_audio}: {e}")
            missing_files += 1
    
    # Write CSV file
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['audio', 'text'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"{split_name} split completed!")
    print(f"  - Successfully processed: {len(csv_data)} files")
    print(f"  - Missing audio files: {missing_files}")
    print(f"  - CSV saved to: {csv_file}")

def main():
    print("=" * 60)
    print("Dataset Preparation Script")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Read TSV file
    data = read_tsv_file()
      # Split data
    splits = split_data(data)
    
    # Process each split
    for split_name, data_split in splits.items():
        process_split(split_name, data_split)
    
    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
