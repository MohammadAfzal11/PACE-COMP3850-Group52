#!/usr/bin/env python3
"""
Verify Environment Script
This script checks if all required packages are installed and working correctly.
"""

import sys
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20s} installed (version: {version})")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} NOT installed")
        return False

def main():
    """Main verification function"""
    print("=" * 60)
    print("Environment Verification for PACE-COMP3850-Group52")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    py_version = sys.version_info
    if py_version >= (3, 8):
        print("✓ Python version is compatible (3.8+)")
    else:
        print("✗ Python version is too old (need 3.8+)")
        return False
    
    print()
    print("-" * 60)
    print("Checking required packages...")
    print("-" * 60)
    
    # List of required packages
    packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('tensorflow', 'tensorflow'),
        ('bitarray', 'bitarray'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('jupyter', 'jupyter'),
    ]
    
    all_ok = True
    missing_packages = []
    
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_ok = False
            missing_packages.append(package_name)
    
    print()
    print("-" * 60)
    
    if all_ok:
        print("✅ All packages are installed correctly!")
        print()
        print("You can now run:")
        print("  - Jupyter notebooks: jupyter notebook")
        print("  - Python scripts: python <script_name>.py")
    else:
        print("⚠️  Some packages are missing:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print()
        print("To install missing packages, run:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
