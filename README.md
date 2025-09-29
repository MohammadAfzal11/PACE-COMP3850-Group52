# PACE-COMP3850-Group52
Cyber Security Defence Stream

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

### Running Notebooks
```bash
cd notebooks/
# For PPRL (original provided code):
jupyter notebook PPRL.ipynb

# For new FPN-RL system:
jupyter notebook federated_embedding_linkage.ipynb
```

### Running Python Scripts
```bash
cd python_files/
# Demo of FPN-RL system:
python demo_fpn_rl.py

# Comparative evaluation:
python comparative_evaluation.py
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
