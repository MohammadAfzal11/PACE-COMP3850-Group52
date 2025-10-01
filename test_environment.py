#!/usr/bin/env python3
"""
Quick Test Script - PACE-COMP3850-Group52
This script performs a quick test to verify the environment is working correctly.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("=" * 70)
    print("Testing Package Imports")
    print("=" * 70)
    print()
    
    tests = []
    
    # Test basic packages
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        tests.append(False)
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        tests.append(False)
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
        tests.append(False)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ TensorFlow: {e}")
        tests.append(False)
    
    try:
        from bitarray import bitarray
        print(f"✓ bitarray")
        tests.append(True)
    except ImportError as e:
        print(f"✗ bitarray: {e}")
        tests.append(False)
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        tests.append(False)
    
    print()
    return all(tests)


def test_csv_files():
    """Test if CSV files are accessible"""
    print("=" * 70)
    print("Testing CSV Files")
    print("=" * 70)
    print()
    
    csv_dir = 'csv_files'
    if not os.path.exists(csv_dir):
        print(f"✗ CSV directory '{csv_dir}' not found")
        return False
    
    required_files = [
        'Alice_numrec_100_corr_25.csv',
        'Bob_numrec_100_corr_25.csv',
    ]
    
    tests = []
    for filename in required_files:
        filepath = os.path.join(csv_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
            tests.append(True)
        else:
            print(f"✗ {filename} not found")
            tests.append(False)
    
    print()
    return all(tests)


def test_python_modules():
    """Test if Python modules are accessible"""
    print("=" * 70)
    print("Testing Python Modules")
    print("=" * 70)
    print()
    
    # Add python_files to path
    sys.path.insert(0, 'python_files')
    
    tests = []
    
    try:
        from BF import BF
        print("✓ BF module can be imported")
        tests.append(True)
    except ImportError as e:
        print(f"✗ BF module: {e}")
        tests.append(False)
    
    try:
        from PPRL import Link
        print("✓ PPRL module can be imported")
        tests.append(True)
    except ImportError as e:
        print(f"✗ PPRL module: {e}")
        tests.append(False)
    
    try:
        from federated_embedding_linkage import FederatedEmbeddingLinkage
        print("✓ federated_embedding_linkage module can be imported")
        tests.append(True)
    except ImportError as e:
        print(f"✗ federated_embedding_linkage module: {e}")
        tests.append(False)
    
    print()
    return all(tests)


def test_standalone_scripts():
    """Test if standalone scripts exist and have valid syntax"""
    print("=" * 70)
    print("Testing Standalone Scripts")
    print("=" * 70)
    print()
    
    scripts = [
        'Afzal/Siamese_CBF_Linkage.py',
        'Afzal/Differential_Privacy_CBF_Linkage.py',
        'Afzal/Federated_Embedding_Linkage.py',
    ]
    
    tests = []
    for script in scripts:
        if os.path.exists(script):
            # Check syntax
            try:
                import py_compile
                py_compile.compile(script, doraise=True)
                print(f"✓ {script} - syntax valid")
                tests.append(True)
            except py_compile.PyCompileError as e:
                print(f"✗ {script} - syntax error: {e}")
                tests.append(False)
        else:
            print(f"✗ {script} - file not found")
            tests.append(False)
    
    print()
    return all(tests)


def main():
    """Main test function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PACE-COMP3850-Group52 - Environment Test".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    results = []
    
    # Run all tests
    results.append(("Package Imports", test_imports()))
    results.append(("CSV Files", test_csv_files()))
    results.append(("Python Modules", test_python_modules()))
    results.append(("Standalone Scripts", test_standalone_scripts()))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print()
        print("Your environment is ready to use!")
        print()
        print("Next steps:")
        print("  1. Run standalone scripts: cd Afzal/ && python Federated_Embedding_Linkage.py")
        print("  2. Start Jupyter: jupyter notebook")
        print("  3. See QUICKSTART.md for more information")
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("Please check the errors above and:")
        print("  1. Make sure you activated the virtual environment:")
        print("     source venv/bin/activate  (Linux/Mac)")
        print("     venv\\Scripts\\activate     (Windows)")
        print()
        print("  2. Install missing packages:")
        print("     pip install -r requirements.txt")
        print()
        print("  3. Run this test again:")
        print("     python test_environment.py")
    
    print("=" * 70)
    print()
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
