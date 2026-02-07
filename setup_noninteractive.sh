#!/bin/bash

# Non-Interactive Setup Script for Vast.ai
# This script sets up the environment without user prompts

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
echo -e "${BLUE}║${NC}  ${GREEN}Non-Interactive Setup for Vast.ai${NC}          ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Validate environment variables
print_step "Step 1/6: Validating environment variables..."

if [ -z "$HF_TOKEN" ]; then
    print_error "HF_TOKEN environment variable is not set!"
    print_error "This script requires HF_TOKEN to be set in the environment"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    print_error "WANDB_API_KEY environment variable is not set!"
    print_error "This script requires WANDB_API_KEY to be set in the environment"
    exit 1
fi

print_success "Environment variables validated"
echo ""

# Step 2: Update apt library
print_step "Step 2/6: Updating apt package lists..."
if apt update -y; then
    print_success "Package lists updated"
else
    print_error "Failed to update package lists"
    exit 1
fi
echo ""

# Step 3: Install system dependencies
print_step "Step 3/6: Installing system dependencies (ffmpeg, nano)..."
if apt install -y nano ffmpeg; then
    print_success "System dependencies installed"
else
    print_error "Failed to install system dependencies"
    exit 1
fi
echo ""

# Step 4: Handle virtual environment
print_step "Step 4/6: Setting up Python virtual environment..."

VENV_PATH=".venv"

if [ -d "$VENV_PATH" ]; then
    print_warning "Virtual environment already exists at $VENV_PATH"

    # Validate existing venv
    if [ -f "$VENV_PATH/bin/activate" ]; then
        print_success "Existing virtual environment appears valid, keeping it"
    else
        print_warning "Existing virtual environment appears corrupted, recreating..."
        rm -rf "$VENV_PATH"

        if python3 -m venv "$VENV_PATH"; then
            print_success "Virtual environment created at $VENV_PATH"
        else
            print_error "Failed to create virtual environment"
            exit 1
        fi
    fi
else
    # Create new venv
    print_step "Creating new virtual environment..."
    if python3 -m venv "$VENV_PATH"; then
        print_success "Virtual environment created at $VENV_PATH"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi
echo ""

# Step 5: Install Python dependencies
print_step "Step 5/6: Installing Python dependencies..."

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip
print_step "Upgrading pip..."
if pip install --upgrade pip; then
    print_success "pip upgraded"
else
    print_warning "Failed to upgrade pip, continuing anyway..."
fi

# Install requirements
print_step "Installing packages from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

if pip install -r requirements.txt; then
    print_success "All packages installed successfully"
else
    print_error "Failed to install packages from requirements.txt"
    exit 1
fi
echo ""

# Step 6: Verify critical packages
print_step "Step 6/6: Verifying critical packages..."

PACKAGES=("transformers" "torch" "datasets" "accelerate" "evaluate" "peft" "wandb")
ALL_OK=true

for package in "${PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $package"
    else
        echo -e "  ${RED}✗${NC} $package"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = true ]; then
    print_success "All critical packages verified"
else
    print_warning "Some packages could not be imported, but installation completed"
fi
echo ""

# Final success message
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${GREEN}Setup completed successfully!${NC}                ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""
print_success "Environment is ready for training"
print_success "Virtual environment: $VENV_PATH"
print_success "To activate manually: source $VENV_PATH/bin/activate"
