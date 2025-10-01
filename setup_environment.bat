@echo off
REM Setup Environment Script for PACE-COMP3850-Group52 (Windows)
REM This script creates a Python virtual environment and installs all dependencies

echo ==================================================
echo PACE-COMP3850-Group52 Environment Setup (Windows)
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

echo.
echo Step 1: Creating virtual environment...
echo ----------------------------------------

REM Check if venv already exists
if exist "venv\" (
    echo [WARNING] Virtual environment 'venv' already exists.
    set /p REPLY="Do you want to delete it and create a new one? (Y/N): "
    if /i "%REPLY%"=="Y" (
        rmdir /s /q venv
        echo [OK] Removed existing virtual environment
    ) else (
        echo Using existing virtual environment
    )
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully
)

echo.
echo Step 2: Activating virtual environment...
echo ----------------------------------------
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

echo.
echo Step 3: Upgrading pip...
echo ----------------------------------------
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip (continuing anyway)
) else (
    echo [OK] Pip upgraded successfully
)

echo.
echo Step 4: Installing dependencies from requirements.txt...
echo ----------------------------------------
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install some dependencies
    echo Please check the error messages above and try again.
    pause
    exit /b 1
)
echo [OK] All dependencies installed successfully

echo.
echo Step 5: Verifying installation...
echo ----------------------------------------

REM Verify key packages
python -c "import numpy" 2>nul && (echo [OK] numpy installed) || (echo [ERROR] numpy NOT installed)
python -c "import pandas" 2>nul && (echo [OK] pandas installed) || (echo [ERROR] pandas NOT installed)
python -c "import sklearn" 2>nul && (echo [OK] sklearn installed) || (echo [ERROR] sklearn NOT installed)
python -c "import tensorflow" 2>nul && (echo [OK] tensorflow installed) || (echo [ERROR] tensorflow NOT installed)
python -c "import bitarray" 2>nul && (echo [OK] bitarray installed) || (echo [ERROR] bitarray NOT installed)
python -c "import matplotlib" 2>nul && (echo [OK] matplotlib installed) || (echo [ERROR] matplotlib NOT installed)
python -c "import jupyter" 2>nul && (echo [OK] jupyter installed) || (echo [ERROR] jupyter NOT installed)

echo.
echo ==================================================
echo [SUCCESS] Setup completed!
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate
echo.
echo To run Jupyter notebooks:
echo   cd notebooks
echo   jupyter notebook
echo.
echo To run Python scripts:
echo   cd python_files
echo   python demo_fpn_rl.py
echo.
echo To deactivate the environment when done:
echo   deactivate
echo ==================================================
echo.
pause
