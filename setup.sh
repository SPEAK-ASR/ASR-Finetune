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

# Step 4: Create Python virtual environment
print_step "Step 4/6: Setting up Python virtual environment..."

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        python -m venv "$VENV_DIR"
        print_success "Virtual environment recreated"
    else
        print_warning "Using existing virtual environment"
    fi
else
    python -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
fi
echo ""

# Step 5: Activate virtual environment and install packages
print_step "Step 5/6: Installing Python packages from requirements.txt..."

# Activate virtual environment
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

# Install packages
print_step "Installing packages (this may take several minutes)..."
if pip install -r requirements.txt; then
    print_success "All packages installed successfully"
else
    print_error "Failed to install some packages"
    exit 1
fi
echo ""

