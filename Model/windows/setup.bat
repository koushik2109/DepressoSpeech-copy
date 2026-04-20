@echo off
REM DepressoSpeech - Environment Setup (Windows)

echo ==========================================
echo   DepressoSpeech - Environment Setup
echo ==========================================

cd /d "%~dp0\.."

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/4] Upgrading pip...
pip install --upgrade pip

echo [3/4] Installing dependencies...
pip install -r requirements.txt

echo [4/4] Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\features" mkdir data\features
if not exist "data\splits" mkdir data\splits
if not exist "checkpoints\scalers" mkdir checkpoints\scalers
if not exist "logs" mkdir logs

echo.
echo ==========================================
echo   Setup complete!
echo   Activate env: .venv\Scripts\activate.bat
echo ==========================================
pause
