@echo off
REM DepressoSpeech - Batch Inference (Windows)
REM Predicts PHQ-8 scores for all audio files in a directory

echo ==========================================
echo   DepressoSpeech - Batch Inference
echo ==========================================

cd /d "%~dp0\.."
call .venv\Scripts\activate.bat

if "%CONFIG%"=="" set CONFIG=configs\inference_config.yaml
set AUDIO_DIR=%1
set OUTPUT_CSV=%2

if "%AUDIO_DIR%"=="" (
    echo Usage: windows\predict_batch.bat ^<audio_directory^> [output.csv]
    echo.
    echo Example:
    echo   windows\predict_batch.bat data\test_audio\
    echo   windows\predict_batch.bat data\test_audio\ results.csv
    exit /b 1
)

if not exist "%AUDIO_DIR%" (
    echo ERROR: Directory not found: %AUDIO_DIR%
    exit /b 1
)

if not exist "checkpoints\best_model.pt" (
    echo ERROR: Model checkpoint not found at checkpoints\best_model.pt
    echo Please train the model first: windows\train.bat
    exit /b 1
)

echo Config:    %CONFIG%
echo Audio Dir: %AUDIO_DIR%
echo.
echo Running batch inference...
echo ------------------------------------------

if "%OUTPUT_CSV%"=="" (
    python scripts\predict.py --audio-dir %AUDIO_DIR% --config %CONFIG%
) else (
    python scripts\predict.py --audio-dir %AUDIO_DIR% --config %CONFIG% --output %OUTPUT_CSV%
)

echo ------------------------------------------
echo   Batch inference complete!
echo ==========================================
pause
