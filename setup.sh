#!/bin/bash

# Exit on any error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

print_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

# Print banner
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${GREEN}Whisper ASR Fine-Tuning Setup Script${NC}        ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Update apt library
print_step "Step 1/6: Updating apt package lists..."
if apt update -y; then
    print_success "Package lists updated"
else
    print_error "Failed to update package lists"
    exit 1
fi
echo ""

# Step 2: Install nano and ffmpeg
print_step "Step 2/6: Installing nano and ffmpeg..."
if apt install -y nano ffmpeg; then
    print_success "nano and ffmpeg installed"
else
    print_error "Failed to install packages"
    exit 1
fi
echo ""

# Step 3: Create .env file with HF token
print_step "Step 3/6: Setting up .env file with HuggingFace token..."

# Check if .env already exists
if [ -f ".env" ]; then
    print_warning ".env file already exists"
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Keeping existing .env file"
    else
        rm .env
        print_warning "Removed old .env file"
    fi
fi

# Prompt for tokens if .env doesn't exist or was removed
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Please enter your HuggingFace token:${NC}"
    echo -e "${YELLOW}(You can get it from: https://huggingface.co/settings/tokens)${NC}"
    read -r HF_TOKEN

    if [ -z "$HF_TOKEN" ]; then
        print_error "No HuggingFace token provided. Exiting."
        exit 1
    fi

    echo -e "${YELLOW}Please enter your Weights & Biases API key:${NC}"
    echo -e "${YELLOW}(You can get it from: https://wandb.ai/authorize)${NC}"
    read -r WANDB_API_KEY

    if [ -z "$WANDB_API_KEY" ]; then
        print_error "No W&B API key provided. Exiting."
        exit 1
    fi

    cat <<EOF > .env
HF_TOKEN="$HF_TOKEN"
WANDB_API_KEY="$WANDB_API_KEY"
EOF

    print_success ".env file created with HuggingFace and W&B tokens"
else
    print_success "Using existing .env file"
fi

echo ""