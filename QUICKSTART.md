# Quick Start Guide - PACE-COMP3850-Group52

## Privacy-Preserving Record Linkage Models

This guide will help you set up the environment and run all the linkage models in this repository.

---

## 📋 Prerequisites

- **Python 3.8 or higher** (Python 3.8, 3.9, 3.10, 3.11, or 3.12)
- **Git** (for cloning the repository)
- **At least 4GB of free disk space**
- **Internet connection** (for downloading dependencies)

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Clone the Repository (if not already done)

```bash
git clone https://github.com/MohammadAfzal11/PACE-COMP3850-Group52.git
cd PACE-COMP3850-Group52
```

### Step 2: Run the Setup Script

**On Linux/Mac:**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

**On Windows:**
```batch
setup_environment.bat
```

This will:
- ✓ Create a virtual environment
- ✓ Install all dependencies (tensorflow, scikit-learn, pandas, etc.)
- ✓ Verify the installation

### Step 3: Activate the Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```batch
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

---

## 🎯 Running the Models

### Model 1: Siamese Neural Network + Counting Bloom Filter (Afzal)

**Best for: Balanced performance with privacy guarantees**

```bash
cd Afzal/
python Siamese_CBF_Linkage.py
```

**Expected Results:**
- Accuracy: ~90%
- F1 Score: ~90%
- Training time: ~10 seconds (500 records)
- Output: Training plots and JSON results

---

### Model 2: Differential Privacy + Counting Bloom Filter (Afzal)

**Best for: High privacy with formal guarantees**

```bash
cd Afzal/
python Differential_Privacy_CBF_Linkage.py
```

**Expected Results:**
- Accuracy: 94-97% (depending on epsilon)
- F1 Score: 94-97%
- Privacy: ε=0.5 to ε=2.0 options
- Training time: ~2 seconds

---

### Model 3: Federated Embedding Linkage (NEW)

**Best for: Neural network approach with differential privacy**

```bash
cd Afzal/
python Federated_Embedding_Linkage.py
```

**Expected Results:**
- Accuracy: 85-95%
- F1 Score: 85-95%
- Privacy: (ε, δ)-differential privacy
- Output: Training history plots and results

---

### Model 4: Interactive Jupyter Notebooks

**Best for: Interactive exploration and experimentation**

```bash
# From the repository root
jupyter notebook
```

Then navigate to:
- `notebooks/PPRL.ipynb` - Original Bloom Filter PPRL
- `notebooks/federated_embedding_linkage.ipynb` - FPN-RL interactive
- `notebooks/SNN Final.ipynb` - Siamese Neural Network

---

## 📊 Understanding the Results

### Metrics Explained

- **Accuracy**: Overall correctness of predictions (0-100%)
- **Precision**: Of predicted matches, how many are correct?
- **Recall**: Of actual matches, how many did we find?
- **F1 Score**: Harmonic mean of precision and recall

### Privacy Metrics

- **Epsilon (ε)**: Privacy budget (lower = more private)
  - ε = 0.5: Very high privacy
  - ε = 1.0: High privacy (recommended)
  - ε = 2.0: Medium privacy
  
- **Delta (δ)**: Probability of privacy breach (typically 1e-5)

### Output Files

Each model generates:
- **PNG plots**: Training history visualizations
- **JSON files**: Numerical results
- **Console output**: Detailed metrics and confusion matrices

---

## 🔧 Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
# Make sure you activated the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# If still not working, reinstall
pip install tensorflow
```

### Issue: "No module named 'bitarray'"

**Solution:**
```bash
pip install bitarray
```

### Issue: "CSV file not found"

**Solution:**
Make sure you're running the scripts from the correct directory:
```bash
# For Afzal models
cd Afzal/
python Siamese_CBF_Linkage.py

# The script will look for CSV files in ../csv_files/
```

### Issue: Import errors in Jupyter notebooks

**Solution:**
1. Make sure the virtual environment is activated
2. Install the Jupyter kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv
```
3. In Jupyter, select Kernel → Change Kernel → venv

### Issue: Windows script execution policy error

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📦 Dataset Information

### Available Datasets (in csv_files/)

- **100 records, 25% correlation**: Alice_numrec_100_corr_25.csv, Bob_numrec_100_corr_25.csv
- **100 records, 50% correlation**: Alice_numrec_100_corr_50.csv, Bob_numrec_100_corr_50.csv
- **500 records, 25% correlation**: Alice_numrec_500_corr_25.csv, Bob_numrec_500_corr_25.csv
- **500 records, 50% correlation**: Alice_numrec_500_corr_50.csv, Bob_numrec_500_corr_50.csv

### Dataset Fields

- `rec_id`: Record identifier
- `first_name`: First name
- `last_name`: Last name
- `city`: City
- Additional fields may include: `age`, `postcode`, `state`, etc.

---

## 🎓 Model Comparison

| Model | Accuracy | Privacy | Speed | Best For |
|-------|----------|---------|-------|----------|
| Siamese + CBF | 90% | Medium | Fast | Balanced performance |
| DP + CBF | 95% | High | Very Fast | Maximum privacy |
| Federated Embedding | 90% | High | Medium | Neural learning |
| Bloom Filter PPRL | 85% | Medium | Fast | Baseline approach |

---

## 🔄 Updating the Environment

If new dependencies are added:

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac

# Update packages
pip install -r requirements.txt --upgrade
```

---

## 🧪 Running Tests

To verify everything is working:

```bash
# Test imports
python -c "import numpy, pandas, tensorflow, sklearn, bitarray; print('All imports successful!')"

# Test TensorFlow GPU (optional)
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## 📝 Deactivating the Environment

When you're done:

```bash
deactivate
```

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify Python version: `python --version` (should be 3.8+)
3. Verify virtual environment is activated (you should see `(venv)`)
4. Try reinstalling: Delete `venv/` folder and run setup script again
5. Check the README.md for additional information

---

## 📚 Additional Resources

- **TensorFlow**: https://www.tensorflow.org/
- **Scikit-learn**: https://scikit-learn.org/
- **Differential Privacy**: https://en.wikipedia.org/wiki/Differential_privacy
- **Record Linkage**: https://en.wikipedia.org/wiki/Record_linkage

---

## ✅ Success Checklist

After setup, you should be able to:

- [ ] Activate the virtual environment
- [ ] Import all required libraries without errors
- [ ] Run at least one standalone Python script successfully
- [ ] Open and run Jupyter notebooks
- [ ] See output files (PNG plots, JSON results)

---

**Last Updated**: 2024  
**Maintained By**: PACE-COMP3850-Group52
