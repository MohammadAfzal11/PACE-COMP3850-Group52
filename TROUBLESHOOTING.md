# Troubleshooting Guide - PACE-COMP3850-Group52

This guide provides solutions to common issues you might encounter while setting up and running the privacy-preserving record linkage models.

---

## Installation Issues

### Issue: "python3: command not found"

**Problem**: Python 3 is not installed or not in PATH.

**Solution**:
- **Linux/Ubuntu**: `sudo apt-get install python3 python3-pip python3-venv`
- **Mac**: `brew install python3` (requires Homebrew)
- **Windows**: Download from https://www.python.org/downloads/ and check "Add to PATH" during installation

### Issue: "pip: command not found"

**Problem**: pip is not installed.

**Solution**:
```bash
# Linux/Mac
python3 -m ensurepip --default-pip

# Windows
python -m ensurepip --default-pip
```

### Issue: "No module named 'venv'"

**Problem**: venv module is not available.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install python3-venv

# For other systems, venv should be included with Python 3.3+
```

---

## Dependency Issues

### Issue: "No module named 'tensorflow'"

**Problem**: TensorFlow is not installed or virtual environment is not activated.

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install TensorFlow
pip install tensorflow

# If you have GPU support, you can try:
pip install tensorflow-gpu
```

### Issue: "No module named 'bitarray'"

**Problem**: bitarray package is not installed.

**Solution**:
```bash
pip install bitarray

# If that fails, try:
pip install bitarray --no-cache-dir
```

### Issue: TensorFlow GPU not detected

**Problem**: TensorFlow is not using GPU acceleration.

**Solution**:
```python
# Check GPU availability
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# If no GPU is shown:
# - Make sure CUDA and cuDNN are installed (for NVIDIA GPUs)
# - TensorFlow CPU version works fine for this project
```

### Issue: "Failed to install scikit-learn"

**Problem**: Compilation errors during scikit-learn installation.

**Solution**:
```bash
# Make sure you have build tools
# Linux:
sudo apt-get install build-essential python3-dev

# Mac:
xcode-select --install

# Then retry:
pip install scikit-learn
```

---

## Runtime Issues

### Issue: "FileNotFoundError: 'Alice_numrec_100_corr_25.csv'"

**Problem**: Script cannot find CSV files.

**Solution**:
```bash
# Make sure you're running from the correct directory

# For Afzal scripts:
cd Afzal/
python Federated_Embedding_Linkage.py

# For notebooks:
cd notebooks/
jupyter notebook

# The scripts expect CSV files in ../csv_files/
```

### Issue: "ImportError: cannot import name 'BF'"

**Problem**: Python cannot find the BF module.

**Solution**:

For notebooks, add this at the beginning:
```python
import sys
import os
sys.path.insert(0, os.path.abspath('../python_files'))

# Now import will work
from BF import BF
from PPRL import Link
```

For scripts, make sure you're in the correct directory.

### Issue: Jupyter notebook kernel crashes

**Problem**: Kernel dies when running code.

**Solution**:
```bash
# Reinstall ipykernel
pip install --upgrade ipykernel

# Register the kernel
python -m ipykernel install --user --name=venv

# In Jupyter: Kernel → Change Kernel → venv
```

### Issue: "Permission denied" when running setup script

**Problem**: Script is not executable.

**Solution**:
```bash
# Linux/Mac
chmod +x setup_environment.sh
./setup_environment.sh

# Windows - run in Command Prompt (not PowerShell):
setup_environment.bat
```

---

## Memory Issues

### Issue: "ResourceExhaustedError: OOM when allocating tensor"

**Problem**: Not enough memory for TensorFlow operations.

**Solution**:
```python
# Reduce batch size in the scripts
# For example, in Federated_Embedding_Linkage.py, change:
BATCH_SIZE = 8  # Instead of 32

# Or limit GPU memory growth:
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Issue: System freezes during training

**Problem**: Too much memory usage.

**Solution**:
- Close other applications
- Reduce batch size and number of training pairs
- Use smaller datasets (100 records instead of 500)

---

## Notebook Issues

### Issue: Notebooks show "Kernel starting..." forever

**Problem**: Jupyter kernel is not properly configured.

**Solution**:
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall Jupyter
pip install --upgrade jupyter ipykernel

# Clear Jupyter cache
jupyter notebook --clear

# Restart Jupyter
jupyter notebook
```

### Issue: "ModuleNotFoundError" in notebook but not in terminal

**Problem**: Notebook is using a different Python environment.

**Solution**:
```bash
# Install ipykernel in your virtual environment
pip install ipykernel

# Add the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=pace-venv --display-name="Python (PACE)"

# In Jupyter: Kernel → Change Kernel → Python (PACE)
```

### Issue: Cell output is truncated

**Problem**: Jupyter limits output display.

**Solution**:
```python
# Add at the beginning of notebook
from IPython.core.display import display, HTML
display(HTML("<style>pre { max-height: 400px; overflow-y: auto; }</style>"))

# Or increase limit:
import sys
sys.maxsize = 2**63 - 1
```

---

## Windows-Specific Issues

### Issue: "execution of scripts is disabled on this system"

**Problem**: PowerShell execution policy prevents script execution.

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use Command Prompt instead of PowerShell
```

### Issue: Long file paths on Windows

**Problem**: Windows path length limit (260 characters).

**Solution**:
- Enable long paths: https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
- Or move repository closer to C:\ (e.g., C:\PACE)

### Issue: "venv\Scripts\activate" not found

**Problem**: Virtual environment not created properly.

**Solution**:
```batch
# Delete and recreate
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
```

---

## macOS-Specific Issues

### Issue: "SSL: CERTIFICATE_VERIFY_FAILED"

**Problem**: macOS SSL certificate issue with pip.

**Solution**:
```bash
# Run the Install Certificates command
/Applications/Python\ 3.x/Install\ Certificates.command

# Or temporarily bypass (not recommended for production):
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

### Issue: "xcrun: error: invalid active developer path"

**Problem**: Command line tools not installed.

**Solution**:
```bash
xcode-select --install
```

---

## Performance Issues

### Issue: Training is very slow

**Problem**: CPU-only training without optimization.

**Solutions**:
1. Reduce dataset size (use 100 records instead of 500)
2. Reduce number of epochs (use 20 instead of 50)
3. Reduce batch size if memory allows
4. Use GPU if available (TensorFlow will auto-detect)

### Issue: Script seems to hang

**Problem**: Script is waiting for input or long computation.

**Solution**:
- Check if there are any prompts in the terminal
- Wait a bit - first run may download models/data
- Check CPU usage to see if script is actually running
- Add verbose output to see progress

---

## Verification Steps

### Step 1: Verify Python

```bash
python3 --version  # Should be 3.8 or higher
```

### Step 2: Verify Virtual Environment

```bash
which python  # Should point to venv/bin/python
pip list      # Should show installed packages
```

### Step 3: Verify Imports

```python
python3 -c "import numpy, pandas, tensorflow, sklearn, bitarray; print('Success!')"
```

### Step 4: Run Test Script

```bash
python test_environment.py
```

All tests should pass!

---

## Still Having Issues?

If none of the above solutions work:

1. **Check the logs**: Look for error messages in terminal output
2. **Clean install**: Delete `venv/` directory and run setup script again
3. **System requirements**: Make sure you have at least 4GB RAM and 5GB disk space
4. **Python version**: Try with Python 3.8, 3.9, or 3.10 (3.11+ may have compatibility issues)
5. **Report the issue**: Create an issue on GitHub with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

---

## Quick Fixes Summary

```bash
# Complete clean reinstall
rm -rf venv/
./setup_environment.sh
source venv/bin/activate
python test_environment.py

# Fix import issues
pip install -r requirements.txt --upgrade

# Fix Jupyter kernel
pip install --upgrade jupyter ipykernel
python -m ipykernel install --user --name=venv

# Verify everything
python verify_environment.py
python test_environment.py
```

---

## Performance Tips

1. **Use smaller datasets for testing**: Start with 100 records
2. **Reduce training epochs**: Use 20-30 epochs for quick tests
3. **Close other applications**: Free up memory
4. **Use standalone scripts**: They're faster than notebooks
5. **Run on better hardware**: If available, use a machine with GPU

---

**Last Updated**: 2024  
**For more help**: See QUICKSTART.md or README.md
