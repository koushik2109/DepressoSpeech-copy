@echo off
REM DepressoSpeech - Feature Extraction Pipeline (Windows)
REM Extracts eGeMAPS, MFCC, and text embeddings from raw audio

echo ==========================================
echo   DepressoSpeech - Feature Extraction
echo ==========================================

cd /d "%~dp0\.."
call .venv\Scripts\activate.bat

if "%DATA_DIR%"=="" set DATA_DIR=data\raw
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=data\features
if "%SPLITS_DIR%"=="" set SPLITS_DIR=data\splits

REM Verify split CSVs
if not exist "%SPLITS_DIR%\train_split.csv" (
    echo ERROR: %SPLITS_DIR%\train_split.csv not found!
    exit /b 1
)
if not exist "%SPLITS_DIR%\dev_split.csv" (
    echo ERROR: %SPLITS_DIR%\dev_split.csv not found!
    exit /b 1
)
if not exist "%SPLITS_DIR%\test_split.csv" (
    echo ERROR: %SPLITS_DIR%\test_split.csv not found!
    exit /b 1
)

echo Configuration:
echo   Data Dir:    %DATA_DIR%
echo   Output Dir:  %OUTPUT_DIR%
echo   Splits Dir:  %SPLITS_DIR%
echo.

echo Starting feature extraction...
echo ------------------------------------------
python scripts\extract_features.py --data-dir %DATA_DIR% --output-dir %OUTPUT_DIR% --splits train dev test --splits-dir %SPLITS_DIR% --id-column Participant_ID

echo ------------------------------------------
echo   Feature extraction complete!
echo   Features saved to: %OUTPUT_DIR%\
echo ==========================================
pause
