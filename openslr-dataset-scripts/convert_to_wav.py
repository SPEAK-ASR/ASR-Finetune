import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

"""
Optional script to convert FLAC audio files to WAV format.
Requires FFmpeg to be installed on your system.

Usage:
    python convert_to_wav.py
"""

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / os.getenv('OUTPUT_DIR', 'sinhala-whisper')
AUDIO_DIR = OUTPUT_DIR / "audio"

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_flac_to_wav(flac_path):
    """Convert a single FLAC file to WAV using FFmpeg"""
    wav_path = flac_path.with_suffix('.wav')
    
    try:
        # Convert to 16kHz mono WAV (Whisper's preferred format)
        subprocess.run([
            'ffmpeg', '-i', str(flac_path),
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # mono
            '-y',            # overwrite output file if exists
            str(wav_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Remove the original FLAC file
        flac_path.unlink()
        
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {flac_path}: {e}")
        return None

def update_csv_files():
    """Update CSV files to reference WAV files instead of FLAC"""
    import csv
    
    data_dir = OUTPUT_DIR / "data"
    
    for csv_file in data_dir.glob("*.csv"):
        print(f"Updating {csv_file.name}...")
        
        # Read existing data
        rows = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Update file extension
                row['audio'] = row['audio'].replace('.flac', '.wav')
                rows.append(row)
        
        # Write updated data
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['audio', 'text']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def main():
    print("=" * 60)
    print("FLAC to WAV Conversion Script")
    print("=" * 60)
    
    # Check if FFmpeg is installed
    if not check_ffmpeg():
        print("\nError: FFmpeg is not installed or not in PATH!")
        print("Please install FFmpeg first:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - Linux: sudo apt-get install ffmpeg")
        print("  - Mac: brew install ffmpeg")
        return
    
    print("\nFFmpeg found! Starting conversion...")
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        split_dir = AUDIO_DIR / split
        if not split_dir.exists():
            print(f"\nWarning: {split_dir} does not exist, skipping...")
            continue
        
        print(f"\nConverting {split} split...")
        flac_files = list(split_dir.glob("*.flac"))
        
        for flac_file in tqdm(flac_files, desc=f"Converting {split}"):
            convert_flac_to_wav(flac_file)
    
    # Update CSV files
    print("\nUpdating CSV files...")
    update_csv_files()
    
    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("=" * 60)
    print("\nAll FLAC files have been converted to 16kHz mono WAV format.")
    print("CSV files have been updated with new file extensions.")

if __name__ == "__main__":
    main()
