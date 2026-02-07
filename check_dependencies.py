"""
Dependency Checker for ASR Training Environment
Validates that all required packages and environment settings are correct
"""

import sys
import os

# Color codes for terminal output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_success(message):
    print(f"{GREEN}✓{NC} {message}")

def print_error(message):
    print(f"{RED}✗{NC} {message}")

def print_warning(message):
    print(f"{YELLOW}⚠{NC} {message}")

def print_header(message):
    print(f"\n{BLUE}{'='*50}{NC}")
    print(f"{BLUE}{message}{NC}")
    print(f"{BLUE}{'='*50}{NC}\n")

def check_python_version():
    """Check if Python version is >= 3.8"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor}.{version.micro} is too old (need >= 3.8)")
        return False

def check_venv():
    """Check if running in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print_success(f"Virtual environment: Active")
        print_success(f"  Location: {sys.prefix}")
        return True
    else:
        print_warning("Not running in a virtual environment")
        print_warning("  This is not recommended but might be okay in containers")
        return True  # Don't fail, just warn

def check_package(package_name):
    """Check if a package can be imported"""
    try:
        __import__(package_name)
        print_success(f"{package_name:20s} - installed")
        return True
    except ImportError:
        print_error(f"{package_name:20s} - NOT FOUND")
        return False
    except Exception as e:
        print_warning(f"{package_name:20s} - error: {str(e)}")
        return False

def check_environment_variables():
    """Check if required environment variables are set"""
    required_vars = ['HF_TOKEN', 'WANDB_API_KEY']
    all_ok = True

    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Don't print the actual token, just show it's set
            masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
            print_success(f"{var:20s} - set ({masked_value})")
        else:
            print_error(f"{var:20s} - NOT SET")
            all_ok = False

    return all_ok

def main():
    print_header("ASR Training Environment Dependency Check")

    all_checks_passed = True

    # Check Python version
    print("Checking Python version...")
    if not check_python_version():
        all_checks_passed = False

    print()

    # Check virtual environment
    print("Checking virtual environment...")
    if not check_venv():
        all_checks_passed = False

    print()

    # Check environment variables
    print("Checking environment variables...")
    if not check_environment_variables():
        all_checks_passed = False

    print()

    # Check critical packages
    print("Checking critical packages...")
    critical_packages = [
        'transformers',
        'torch',
        'datasets',
        'accelerate',
        'evaluate',
        'jiwer',
        'tensorboard',
        'gradio',
        'peft',
        'wandb',
        'dotenv',
    ]

    for package in critical_packages:
        if not check_package(package):
            all_checks_passed = False

    print()

    # Final summary
    print_header("Summary")

    if all_checks_passed:
        print_success("All checks passed! Environment is ready for training.")
        return 0
    else:
        print_error("Some checks failed. Please review the output above.")
        print_error("You may need to install missing packages or set environment variables.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
