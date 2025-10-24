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

### 🧬 **ash/**
Contains medical record linkage with Content-Based Filtering:
- `working code with random forest and DP.ipynb` - **NEW**: Enhanced medical record linkage with comprehensive feature extraction and CBF
- `working_code_with_random_forest_and_dp.py` - Python version of the notebook
- **Features:**
  - Comprehensive medical information extraction (diagnoses, symptoms, procedures, medications, body parts)
  - Content-Based Filtering (CBF) for medical record matching
  - TF-IDF vectorization for text similarity
  - Differential privacy with Laplace noise
  - Random Forest classifier with privacy-utility tradeoff analysis
  - Multiple privacy budget (epsilon) evaluations

## Key Features

### 1. **Extended Code for Text-Based Encoding and Linkage**
- Support for both structured and unstructured data
- TF-IDF vectorization for text processing
- Mixed-mode feature processing
- **NEW**: Comprehensive medical text parsing and feature extraction

### 2. **Content-Based Filtering (CBF) for Medical Records**
- **NEW**: Extracts medical features from unstructured text:
  - Diagnoses and medical conditions (50+ keywords)
  - Symptoms and clinical presentations (40+ keywords)
  - Medical procedures and interventions (30+ keywords)
  - Medications and drug classes (30+ keywords)
  - Body parts and anatomical locations (35+ keywords)
- Calculates Jaccard similarity for each feature category
- Weighted CBF overall similarity score
- Maintains demographic extraction (age, gender) with multiple pattern matching

### 3. **5 Linkage Models with Threshold-Based Classification**
- Neural network embeddings with privacy guarantees
- Adaptive threshold learning (0.6-0.7 range for non-matches)
- Multiple similarity metrics and approaches
- **NEW**: Random Forest classifier with CBF features

### 3. **Fine-Tuning Capabilities**
- Noise injection for robustness testing
- Dirty data handling mechanisms
- Configurable privacy parameters
- **NEW**: Feature importance analysis for CBF features

### 4. **Differential Privacy Implementation**
- Epsilon (ε) parameter for privacy budget control
- Gaussian noise calibration for embeddings
- Privacy composition tracking
- **NEW**: Laplace noise for medical record features with privacy-utility tradeoff analysis (epsilon values: 10.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1)

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

# For medical record linkage with CBF:
cd ../ash/
jupyter notebook "working code with random forest and DP.ipynb"
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
