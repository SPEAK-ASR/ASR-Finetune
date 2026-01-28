"""
Utility script to clean up HuggingFace cache files and other storage-consuming artifacts.
Run this before training on storage-constrained environments like RunPod.
"""
import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_folder_size(path: str) -> float:
    """Get folder size in GB."""
    if not os.path.exists(path):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(path)
        for filename in filenames
    )
    return total / (1024 ** 3)


def cleanup_hf_cache():
    """Remove HuggingFace dataset cache files."""
    cache_dir = config.HF_DATASETS_CACHE
    logger.info(f"HF Datasets cache: {cache_dir}")
    
    try:
        if os.path.exists(cache_dir):
            size = get_folder_size(cache_dir)
            logger.info(f"  Size: {size:.2f} GB")
            shutil.rmtree(cache_dir)
            logger.info("  ✓ Cleared")
        else:
            logger.info("  Not found (already clean)")
    except Exception as e:
        logger.error(f"  Failed: {str(e)}")


def cleanup_hf_hub_cache():
    """Remove HuggingFace Hub cache (model downloads)."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_cache = os.path.join(hf_home, "hub")
    logger.info(f"HF Hub cache: {hub_cache}")
    
    try:
        if os.path.exists(hub_cache):
            size = get_folder_size(hub_cache)
            logger.info(f"  Size: {size:.2f} GB")
            shutil.rmtree(hub_cache)
            logger.info("  ✓ Cleared")
        else:
            logger.info("  Not found (already clean)")
    except Exception as e:
        logger.error(f"  Failed: {str(e)}")


def cleanup_wandb():
    """Remove WandB local cache."""
    wandb_dirs = ["./wandb", os.path.expanduser("~/.wandb")]
    for wandb_dir in wandb_dirs:
        logger.info(f"WandB cache: {wandb_dir}")
        try:
            if os.path.exists(wandb_dir):
                size = get_folder_size(wandb_dir)
                logger.info(f"  Size: {size:.2f} GB")
                shutil.rmtree(wandb_dir)
                logger.info("  ✓ Cleared")
            else:
                logger.info("  Not found")
        except Exception as e:
            logger.error(f"  Failed: {str(e)}")





def cleanup_old_checkpoints(checkpoint_dir: str = "./checkpoints", keep_last: int = 2):
    """Remove old checkpoints, keeping only the last N."""
    logger.info(f"Checkpoints: {checkpoint_dir}")
    
    try:
        if not os.path.exists(checkpoint_dir):
            logger.info("  Not found")
            return
            
        # Find checkpoint folders (usually named checkpoint-XXXX)
        checkpoints = sorted([
            d for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint-")
        ], key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        
        if len(checkpoints) <= keep_last:
            logger.info(f"  Only {len(checkpoints)} checkpoints, keeping all")
            return
            
        # Remove old checkpoints
        to_remove = checkpoints[:-keep_last]
        for ckpt in to_remove:
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            size = get_folder_size(ckpt_path)
            shutil.rmtree(ckpt_path)
            logger.info(f"  ✓ Removed {ckpt} ({size:.2f} GB)")
            
    except Exception as e:
        logger.error(f"  Failed: {str(e)}")


def cleanup_torch_cache():
    """Remove PyTorch cache."""
    torch_cache = os.path.expanduser("~/.cache/torch")
    logger.info(f"PyTorch cache: {torch_cache}")
    
    try:
        if os.path.exists(torch_cache):
            size = get_folder_size(torch_cache)
            logger.info(f"  Size: {size:.2f} GB")
            shutil.rmtree(torch_cache)
            logger.info("  ✓ Cleared")
        else:
            logger.info("  Not found")
    except Exception as e:
        logger.error(f"  Failed: {str(e)}")


def cleanup_all(keep_checkpoints: int = 2, clean_model_cache: bool = False):
    """
    Clean all caches to free up storage.
    
    Args:
        keep_checkpoints: Number of checkpoints to keep
        clean_model_cache: Whether to clean HF model downloads (will need re-download)
    """
    logger.info("=" * 50)
    logger.info("STORAGE CLEANUP FOR TRAINING")
    logger.info("=" * 50)
    
    cleanup_hf_cache()
    cleanup_wandb()
    cleanup_torch_cache()
    cleanup_old_checkpoints(keep_last=keep_checkpoints)
    
    if clean_model_cache:
        cleanup_hf_hub_cache()
    
    logger.info("=" * 50)
    logger.info("Cleanup complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean up storage for training")
    parser.add_argument("--keep-checkpoints", type=int, default=2, 
                        help="Number of checkpoints to keep")
    parser.add_argument("--clean-models", action="store_true",
                        help="Also clean model downloads (will need re-download)")
    args = parser.parse_args()
    
    cleanup_all(
        keep_checkpoints=args.keep_checkpoints,
        clean_model_cache=args.clean_models
    )