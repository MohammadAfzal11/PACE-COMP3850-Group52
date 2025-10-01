# Notebook Fixes Guide

## Issue 1: PPRL.ipynb - Module Import Errors

### Problem
The PPRL.ipynb notebook has import errors:
```python
from bf import BF    # Error: No module named 'bitarray'
from PPRL import Link
```

### Solution

Before running PPRL.ipynb, you need to:

1. **Install dependencies** (if not already done):
```bash
pip install bitarray matplotlib pandas numpy
```

2. **Add this code at the beginning of the notebook** (in the first cell):

```python
import sys
import os

# Add parent directory to path to import from python_files
sys.path.insert(0, os.path.abspath('../python_files'))
sys.path.insert(0, os.path.abspath('../'))

# Now imports will work
import time
import math

from BF import BF    # Note: Capital letters
from PPRL import Link
```

### Alternative Solution: Copy Files

Copy the required Python files to the notebooks directory:

```bash
# From repository root
cp python_files/BF.py notebooks/
cp python_files/PPRL.py notebooks/
```

Then in the notebook, use:
```python
from BF import BF
from PPRL import Link
```

---

## Issue 2: federated_embedding_linkage.ipynb - All-in-One Approach

### Problem
The notebook tries to define all classes inline, which can be confusing.

### Solution 1: Use the Standalone Script

Instead of using the notebook, use the standalone script:

```bash
cd Afzal/
python Federated_Embedding_Linkage.py
```

This is much simpler and produces the same results!

### Solution 2: Import from Module

Add this at the beginning of the notebook:

```python
import sys
sys.path.insert(0, '../python_files')

from federated_embedding_linkage import FederatedEmbeddingLinkage
```

Then use the imported class instead of defining it in the notebook.

---

## Issue 3: General Import Issues in Notebooks

### Problem
Notebooks can't find modules in parent directories.

### Universal Solution

Add this code block at the **very beginning** of any notebook that has import issues:

```python
# Fix import paths
import sys
import os

# Add parent directory and python_files to path
notebook_dir = os.path.dirname(os.path.abspath('__file__')) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(notebook_dir)
python_files_dir = os.path.join(parent_dir, 'python_files')

sys.path.insert(0, parent_dir)
sys.path.insert(0, python_files_dir)

print(f"Added to path: {parent_dir}")
print(f"Added to path: {python_files_dir}")
```

---

## Quick Reference: Package Installation

If you get "No module named 'X'" errors, install the missing package:

```bash
# Core packages
pip install numpy pandas scipy

# Machine learning
pip install scikit-learn tensorflow

# Bloom filters
pip install bitarray

# Visualization
pip install matplotlib seaborn

# Jupyter
pip install jupyter ipykernel ipywidgets

# Or install everything at once:
pip install -r requirements.txt
```

---

## Testing Your Environment

Run this in a Python cell to verify everything is working:

```python
# Test all imports
try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import sklearn
    from bitarray import bitarray
    import matplotlib.pyplot as plt
    
    print("✓ All imports successful!")
    print(f"  - NumPy: {np.__version__}")
    print(f"  - Pandas: {pd.__version__}")
    print(f"  - TensorFlow: {tf.__version__}")
    print(f"  - Scikit-learn: {sklearn.__version__}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install missing packages:")
    print("  pip install -r requirements.txt")
```

---

## Summary

**Recommended Workflow:**

1. ✅ **Use setup script first**: `./setup_environment.sh` or `setup_environment.bat`
2. ✅ **Activate environment**: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. ✅ **For quick testing**: Use standalone Python scripts in `Afzal/` directory
4. ✅ **For interactive work**: Use Jupyter notebooks with the import fixes above

**Still having issues?**

Run the verification script:
```bash
python verify_environment.py
```

This will check if all packages are installed correctly.
