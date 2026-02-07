# Vast.ai Setup Guide for ASR Training

This guide will help you set up automatic training for your ASR model on Vast.ai bid instances.

## Overview

The automation scripts in this repository allow your ASR training to start automatically when you win a Vast.ai bid, without manual intervention.

**Key Features:**
- ✅ Automatic setup on first run
- ✅ Fast restarts (1-2 minutes)
- ✅ Automatic checkpoint resumption
- ✅ Persistent storage support
- ✅ No manual intervention required

## Prerequisites

1. **HuggingFace Account & Token**
   - Get your token from: https://huggingface.co/settings/tokens
   - Needs `write` access to push models

2. **Weights & Biases Account & API Key**
   - Get your API key from: https://wandb.ai/authorize

3. **Vast.ai Account**
   - Sign up at: https://vast.ai/

## Quick Start

### Step 1: Configure Vast.ai Instance

When creating your Vast.ai instance, use these settings:

**Base Image:**
```
pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
```
(Or any PyTorch official image with Python 3.8+)

**Environment Variables:**
Set these in the Vast.ai dashboard when creating the instance:
```
HF_TOKEN=<your_huggingface_token_here>
WANDB_API_KEY=<your_wandb_api_key_here>
PYTHONUNBUFFERED=1
```

**Disk Space:**
- Minimum: 50 GB
- Recommended: 100 GB (for caching models and datasets)

**GPU Requirements:**
- Must support bf16 (Ampere or newer)
- Recommended: RTX 3090, A5000, A6000, or better

**On-Start Script:**
```bash
bash /workspace/ASR-Finetune/vast_startup.sh
```

### Step 2: Initial Setup (First Time Only)

1. **Create and start your Vast.ai instance** with the settings above

2. **SSH into your instance:**
   ```bash
   ssh root@<your_instance_ip> -p <port>
   ```

3. **Clone your repository:**
   ```bash
   cd /workspace
   git clone <your_repo_url> ASR-Finetune
   cd ASR-Finetune
   ```

4. **Make scripts executable:**
   ```bash
   chmod +x vast_startup.sh
   chmod +x setup_noninteractive.sh
   ```

5. **Stop and restart the instance** from Vast.ai dashboard
   - This triggers the automatic startup script
   - Monitor progress: `tail -f /workspace/vast_startup.log`

### Step 3: Monitor Training

After restart, the container will automatically:
1. Run setup (first time: ~10-15 minutes)
2. Install dependencies
3. Launch training

**Monitor startup:**
```bash
tail -f /workspace/vast_startup.log
```

**Monitor training:**
```bash
tail -f /workspace/ASR-Finetune/logs/training.log
```

**Check if training is running:**
```bash
ps aux | grep python | grep main.py
```

## Directory Structure

Your persistent volume (`/workspace`) should have this structure:

```
/workspace/
├── ASR-Finetune/              # Your git repository
│   ├── .venv/                 # Virtual environment (created automatically)
│   ├── cache/                 # HuggingFace datasets cache
│   ├── models/                # HuggingFace models cache
│   ├── logs/                  # Application logs
│   │   └── training.log       # Training output
│   ├── wandb/                 # Weights & Biases logs
│   ├── checkpoints/           # Training checkpoints (for resumption)
│   ├── vast_startup.sh        # Main startup script
│   ├── setup_noninteractive.sh # Setup script
│   ├── check_dependencies.py  # Dependency validator
│   └── .vastai_setup_complete # Setup flag (auto-created)
└── vast_startup.log           # Startup logs
```

## How It Works

### First Run (Cold Start)
1. Container starts
2. `vast_startup.sh` detects no setup flag
3. Runs `setup_noninteractive.sh`:
   - Installs ffmpeg and system dependencies (2-3 min)
   - Creates Python virtual environment (1-2 min)
   - Installs Python packages from requirements.txt (5-8 min)
4. Creates `.vastai_setup_complete` flag file
5. Activates venv
6. Launches `main.py`
7. Training starts (downloads models/datasets on first run)

**Total time: ~15-20 minutes**

### Restart (Warm Start)
1. Container starts
2. `vast_startup.sh` detects setup flag
3. Validates existing venv (< 10 seconds)
4. Activates venv
5. Launches `main.py`
6. Training resumes from last checkpoint

**Total time: ~1-2 minutes**

### After Preemption (Outbid)
When you get outbid and then win again:
1. Instance resumes (warm start flow)
2. Training automatically resumes from last checkpoint
3. No data loss (checkpoints saved every 3000 steps)

## Configuration

### Training Configuration

The training is configured in `src/config/config.py`. Key settings:

- **Model:** openai/whisper-small
- **Language:** Sinhala
- **Batch size:** 32 (train), 256 (eval)
- **Learning rate:** 3e-5
- **Epochs:** 5.0
- **Checkpointing:** Every 3000 steps
- **Mixed precision:** bf16

### Changing Configuration

To modify training parameters:
1. Edit `src/config/config.py`
2. Commit changes to git
3. Pull latest on Vast.ai instance: `git pull`
4. Restart instance

## Troubleshooting

### Training Not Starting

**Check startup logs:**
```bash
cat /workspace/vast_startup.log
```

**Common issues:**
- Missing environment variables (HF_TOKEN, WANDB_API_KEY)
- Insufficient disk space
- Network connectivity issues

**Solution:** Verify environment variables in Vast.ai dashboard

### Virtual Environment Errors

**Symptom:** Setup fails or venv activation fails

**Solution:**
```bash
cd /workspace/ASR-Finetune
rm -f .vastai_setup_complete  # Force re-setup
rm -rf .venv                   # Remove corrupted venv
# Restart instance
```

### Checkpoint Not Resuming

**Check if checkpoints exist:**
```bash
ls -la /workspace/ASR-Finetune/checkpoints/
```

**Common issues:**
- Checkpoints stored in non-persistent location
- Checkpoints deleted or corrupted

**Solution:** Ensure `/workspace` is mounted as persistent storage in Vast.ai

### Out of Memory Errors

The training config has `auto_find_batch_size=True` which automatically reduces batch size if OOM occurs. If issues persist:

1. Edit `src/config/config.py`
2. Reduce `per_device_train_batch_size` from 32 to 16 or 8
3. Restart training

### Dependency Installation Fails

**Check package installation logs:**
```bash
cat /workspace/vast_startup.log | grep -A 20 "Installing Python dependencies"
```

**Solution:**
```bash
cd /workspace/ASR-Finetune
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Can't Connect to HuggingFace Hub

**Symptom:** Authentication errors or dataset download failures

**Check token:**
```bash
echo $HF_TOKEN | head -c 10  # Shows first 10 chars
```

**Solution:**
1. Verify token in Vast.ai dashboard
2. Ensure token has `write` access
3. Test authentication:
   ```bash
   python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
   ```

## Advanced Usage

### Manual Training Launch

If you need to run training manually:

```bash
cd /workspace/ASR-Finetune
source .venv/bin/activate
python main.py
```

### Debugging Setup

Run setup script manually to see detailed output:

```bash
cd /workspace/ASR-Finetune
export HF_TOKEN="your_token"
export WANDB_API_KEY="your_key"
bash setup_noninteractive.sh
```

### Checking Dependencies

Validate your environment:

```bash
cd /workspace/ASR-Finetune
source .venv/bin/activate
python check_dependencies.py
```

### Viewing Training Metrics

**Weights & Biases Dashboard:**
1. Visit https://wandb.ai
2. Go to your project
3. View real-time metrics, losses, and WER

**TensorBoard (if needed):**
```bash
cd /workspace/ASR-Finetune
source .venv/bin/activate
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

### Stopping Training Gracefully

```bash
# Find training process
ps aux | grep python | grep main.py

# Send interrupt signal (allows checkpoint save)
kill -INT <pid>

# Wait for checkpoint to save (check logs)
tail -f logs/training.log
```

## Best Practices

### 1. Use Persistent Storage
- Always mount `/workspace` as persistent storage
- Verify mount before starting training
- Checkpoints, cache, and logs should be on persistent storage

### 2. Monitor Costs
- Bid instances can be preempted anytime
- Set maximum bid price to control costs
- Use spot instances for cost savings

### 3. Regular Checkpoints
- Default: Every 3000 steps
- Keeps last 5 checkpoints
- Automatic cleanup of old checkpoints

### 4. Branch Protection
- Don't train on `main` branch directly
- Create feature branches for experiments
- Push to Hub uses configured model ID

### 5. Experiment Tracking
- All runs logged to W&B automatically
- Use descriptive run names in config
- Tag experiments for easy comparison

## Useful Commands

### Instance Management
```bash
# Check disk usage
df -h /workspace

# Check GPU usage
nvidia-smi

# Check memory usage
free -h

# Check running processes
ps aux | grep python
```

### Git Management
```bash
# Pull latest code
cd /workspace/ASR-Finetune
git pull

# Check current branch
git branch

# View recent commits
git log --oneline -5
```

### Log Management
```bash
# View startup log
tail -f /workspace/vast_startup.log

# View training log
tail -f /workspace/ASR-Finetune/logs/training.log

# Search for errors
grep -i error /workspace/vast_startup.log
```

## Support

If you encounter issues:

1. Check logs: `vast_startup.log` and `logs/training.log`
2. Verify environment variables are set
3. Ensure persistent storage is mounted
4. Check GPU compatibility (bf16 support)

For code issues, check the main repository documentation.

## Next Steps

After successful setup:

1. Monitor first training run completion
2. Verify model pushed to HuggingFace Hub
3. Check W&B dashboard for metrics
4. Set up alerts for training completion
5. Schedule regular checkpoint backups

Happy training! 🚀
