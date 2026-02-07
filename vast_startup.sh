#!/bin/bash

# Vast.ai Auto-Start Script
# This script automatically sets up and launches ASR training when the container starts

# Exit on any error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Log file location
LOG_FILE="/workspace/vast_startup.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to print colored messages
print_step() {
    echo -e "${BLUE}==>${NC} ${GREEN}$1${NC}" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

print_info() {
    echo -e "${CYAN}INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Print banner
echo "" | tee -a "$LOG_FILE"
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}║${NC}  ${GREEN}Vast.ai ASR Training Auto-Start${NC}            ${BLUE}║${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

log "Starting Vast.ai auto-start script..."
log "Working directory: $SCRIPT_DIR"

# Validate required environment variables
print_step "Validating environment variables..."

if [ -z "$HF_TOKEN" ]; then
    print_error "HF_TOKEN environment variable is not set!"
    print_error "Please set HF_TOKEN in Vast.ai dashboard environment variables"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    print_error "WANDB_API_KEY environment variable is not set!"
    print_error "Please set WANDB_API_KEY in Vast.ai dashboard environment variables"
    exit 1
fi

print_success "Environment variables validated"

# Flag file to track setup completion
SETUP_FLAG=".vastai_setup_complete"

# Check if this is first-time setup or restart
if [ ! -f "$SETUP_FLAG" ]; then
    print_step "First-time setup detected - running full setup..."
    log "Flag file not found: $SETUP_FLAG"

    # Run non-interactive setup
    if [ -f "./setup_noninteractive.sh" ]; then
        print_info "Executing setup_noninteractive.sh..."
        bash ./setup_noninteractive.sh

        if [ $? -eq 0 ]; then
            print_success "Setup completed successfully"

            # Create flag file with metadata
            {
                echo "SETUP_TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
                echo "SETUP_VERSION=1.0"
                echo "VENV_PATH=$SCRIPT_DIR/.venv"
                echo "PYTHON_VERSION=$(python3 --version 2>&1)"
            } > "$SETUP_FLAG"

            print_success "Setup flag created: $SETUP_FLAG"
        else
            print_error "Setup failed! Check logs above for details"
            exit 1
        fi
    else
        print_error "setup_noninteractive.sh not found!"
        exit 1
    fi
else
    print_step "Restart detected - skipping setup..."
    log "Flag file found: $SETUP_FLAG"

    # Display setup metadata
    if [ -f "$SETUP_FLAG" ]; then
        print_info "Setup metadata:"
        cat "$SETUP_FLAG" | while read line; do
            print_info "  $line"
        done
    fi

    # Quick venv validation
    if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
        print_success "Virtual environment verified"
    else
        print_warning "Virtual environment missing or corrupted!"
        print_warning "Deleting setup flag to force re-setup..."
        rm -f "$SETUP_FLAG"
        print_error "Please restart the container to trigger re-setup"
        exit 1
    fi
fi

# Activate virtual environment
print_step "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    print_success "Virtual environment activated"
    log "Python: $(which python)"
    log "Python version: $(python --version 2>&1)"
else
    print_error "Virtual environment activation script not found!"
    exit 1
fi

# Set Python to unbuffered mode for real-time logs
export PYTHONUNBUFFERED=1

# Optional: Run dependency check
if [ -f "check_dependencies.py" ]; then
    print_step "Validating dependencies..."
    if python check_dependencies.py; then
        print_success "All dependencies validated"
    else
        print_warning "Dependency check failed, but continuing anyway..."
    fi
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch training
print_step "Launching ASR training..."
print_info "Training logs will be written to: logs/training.log"
print_info "Monitor with: tail -f logs/training.log"
echo "" | tee -a "$LOG_FILE"

log "Executing: accelerate launch --multi_gpu --num_processes=2 main.py"

# Launch training and capture exit code
accelerate launch --multi_gpu --num_processes=2 main.py 2>&1 | tee -a logs/training.log
EXIT_CODE=${PIPESTATUS[0]}

# Log completion
echo "" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    print_success "Training completed successfully (exit code: $EXIT_CODE)"
    log "Training finished at $(date)"
else
    print_error "Training exited with error code: $EXIT_CODE"
    log "Training failed at $(date)"
fi

exit $EXIT_CODE
