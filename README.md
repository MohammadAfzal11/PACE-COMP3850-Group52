# PACE-COMP3850-Group52
Cyber Security Defence Stream

## Quick Start - Environment Setup

### 🚀 Automated Setup (Recommended)

**For Linux/Mac:**
```bash
./setup_environment.sh
source venv/bin/activate
python test_environment.py  # Verify installation
```

**For Windows:**
```batch
setup_environment.bat
venv\Scripts\activate
python test_environment.py
```

These scripts will:
1. ✓ Create a Python virtual environment
2. ✓ Install all required dependencies automatically
3. ✓ Verify the installation

### 📦 What Gets Installed

The following packages will be installed:
- `numpy`, `pandas`, `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities
- `tensorflow` - Deep learning framework
- `bitarray` - Bloom filter implementation
- `matplotlib`, `seaborn` - Visualization
- `jupyter` - Notebook support

### ✅ Verify Installation

After setup, run the test script:
```bash
python test_environment.py
```

This will verify:
- All packages are installed correctly
- CSV data files are accessible
- Python modules can be imported
- Standalone scripts have valid syntax

### 📖 Documentation

- **[SUMMARY.md](SUMMARY.md)** - Complete overview of setup solution
- **[QUICKSTART.md](QUICKSTART.md)** - Comprehensive quick start guide  
- **[NOTEBOOK_FIXES.md](NOTEBOOK_FIXES.md)** - Solutions for common notebook issues
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Detailed troubleshooting guide
- **[README.md](README.md)** - This file (general information)

### Manual Setup (Advanced Users)

If you prefer manual setup:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_environment.py
```

## Repository Structure

This repository has been reorganized into clean, separate folders for better organization and maintainability:

### 📁 **csv_files/** 
Contains all dataset files (12 files total):
- Alice datasets: `Alice_numrec_100_corr_25.csv`, `Alice_numrec_100_corr_50.csv`, etc.
- Bob datasets: `Bob_numrec_100_corr_25.csv`, `Bob_numrec_100_corr_50.csv`, etc.
- Various correlation levels (25%, 50%) and record counts (100, 500)

### 📓 **notebooks/**
Contains all Jupyter notebooks (3 files):
- `PPRL.ipynb` - Original Privacy-Preserving Record Linkage implementation (main provided code)
- `SNN Final.ipynb` - Siamese Neural Network implementation
- `federated_embedding_linkage.ipynb` - **NEW**: Interactive notebook version of FPN-RL system

### 🐍 **python_files/**
Contains all Python modules (6 files):
- `PPRL.py` - Core PPRL implementation
- `BF.py` - Bloom Filter utilities
- `federated_embedding_linkage.py` - Main FPN-RL implementation
- `comparative_evaluation.py` - Performance comparison tools
- `demo_fpn_rl.py` - Demo script for FPN-RL
- `final_report.py` - Report generation utilities

### 🎯 **Afzal/**
Contains standalone linkage model implementations:
- `Siamese_CBF_Linkage.py` - Siamese Neural Network with Counting Bloom Filter (100 records)
- `Siamese_CBF_500_Linkage.py` - Siamese Neural Network (500 records)
- `Differential_Privacy_CBF_Linkage.py` - Differential Privacy with CBF
- `Federated_Embedding_Linkage.py` - **NEW**: Standalone FPN-RL implementation

## Key Features

### 1. **Extended Code for Text-Based Encoding and Linkage**
- Support for both structured and unstructured data
- TF-IDF vectorization for text processing
- Mixed-mode feature processing

### 2. **5 Linkage Models with Threshold-Based Classification**
- Neural network embeddings with privacy guarantees
- Adaptive threshold learning (0.6-0.7 range for non-matches)
- Multiple similarity metrics and approaches

### 3. **Fine-Tuning Capabilities**
- Noise injection for robustness testing
- Dirty data handling mechanisms
- Configurable privacy parameters

### 4. **Differential Privacy Implementation**
- Epsilon (ε) parameter for privacy budget control
- Gaussian noise calibration for embeddings
- Privacy composition tracking

### 5. **Coalition Rate Analysis**
- Performance metrics across different privacy levels
- Privacy-utility tradeoff evaluation
- Comprehensive comparison framework

## Usage

### Option 1: Standalone Python Scripts (Recommended for Testing)

**Run individual linkage models:**
```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Run Siamese Neural Network with CBF
cd Afzal/
python Siamese_CBF_Linkage.py

# Run Federated Embedding Linkage
python Federated_Embedding_Linkage.py

# Run Differential Privacy CBF
python Differential_Privacy_CBF_Linkage.py
```

### Option 2: Jupyter Notebooks (Interactive)

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/Mac

cd notebooks/
jupyter notebook

# Then open:
# - PPRL.ipynb (for Bloom Filter PPRL)
# - federated_embedding_linkage.ipynb (for FPN-RL system)
# - SNN Final.ipynb (for Siamese Neural Network)
```

### Option 3: Python Module Scripts

```bash
# Activate virtual environment first
cd python_files/

# Demo of FPN-RL system:
python demo_fpn_rl.py

# Comparative evaluation:
python comparative_evaluation.py

# Generate final report:
python final_report.py
```

## File Dependencies

All imports and file paths have been updated to work with the new structure:
- Notebooks automatically import from `../python_files/`
- CSV files are accessed via `../csv_files/`
- Python modules can import each other within the same directory

## Results and Deliverables

Expected outputs include:
- Privacy-utility analysis results
- Performance comparison charts
- Resource usage documentation
- Final evaluation reports

The reorganized structure maintains all original functionality while providing better organization and the new interactive notebook interface for the federated embedding linkage system.
