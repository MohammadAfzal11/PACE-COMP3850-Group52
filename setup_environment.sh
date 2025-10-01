#!/bin/bash

# Setup Environment Script for PACE-COMP3850-Group52
# This script creates a Python virtual environment and installs all dependencies

echo "=================================================="
echo "PACE-COMP3850-Group52 Environment Setup"
echo "=================================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"

# Check Python version (minimum 3.8)
PYTHON_MIN_VERSION="3.8"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8 or higher is required."
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo ""
echo "Step 1: Creating virtual environment..."
echo "----------------------------------------"

# Check if venv already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment 'venv' already exists."
    read -p "Do you want to delete it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "✓ Removed existing virtual environment"
    else
        echo "Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo "✓ Virtual environment created successfully"
    else
        echo "❌ Error: Failed to create virtual environment"
        exit 1
    fi
fi

echo ""
echo "Step 2: Activating virtual environment..."
echo "----------------------------------------"
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

echo ""
echo "Step 3: Upgrading pip..."
echo "----------------------------------------"
pip install --upgrade pip setuptools wheel
if [ $? -eq 0 ]; then
    echo "✓ Pip upgraded successfully"
else
    echo "⚠️  Warning: Failed to upgrade pip (continuing anyway)"
fi

echo ""
echo "Step 4: Installing dependencies from requirements.txt..."
echo "----------------------------------------"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✓ All dependencies installed successfully"
    else
        echo "❌ Error: Failed to install some dependencies"
        echo "Please check the error messages above and try again."
        exit 1
    fi
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

echo ""
echo "Step 5: Verifying installation..."
echo "----------------------------------------"

# Verify key packages
PACKAGES=("numpy" "pandas" "sklearn" "tensorflow" "bitarray" "matplotlib" "jupyter")
ALL_OK=true

for package in "${PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✓ $package installed"
    else
        echo "❌ $package NOT installed"
        ALL_OK=false
    fi
done

echo ""
echo "=================================================="
if [ "$ALL_OK" = true ]; then
    echo "✅ Setup completed successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run Jupyter notebooks:"
    echo "  cd notebooks/"
    echo "  jupyter notebook"
    echo ""
    echo "To run Python scripts:"
    echo "  cd python_files/"
    echo "  python demo_fpn_rl.py"
    echo ""
    echo "To deactivate the environment when done:"
    echo "  deactivate"
else
    echo "⚠️  Setup completed with some errors."
    echo "Please check the messages above and install missing packages manually."
fi
echo "=================================================="
