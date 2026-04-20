@echo off
REM DepressoSpeech - Single File Inference (Windows)
REM Predicts PHQ-8 score from a single audio file

echo ==========================================
echo   DepressoSpeech - Single File Inference
echo ==========================================

cd /d "%~dp0\.."
call .venv\Scripts\activate.bat

if "%CONFIG%"=="" set CONFIG=configs\inference_config.yaml
set AUDIO_FILE=%1

if "%AUDIO_FILE%"=="" (
    echo Usage: windows\predict.bat ^<path_to_audio_file^>
    echo.
    echo Supported formats: .wav .mp3 .flac .ogg .m4a
    echo.
    echo Example:
    echo   windows\predict.bat C:\path\to\interview.wav
    exit /b 1
)

if not exist "%AUDIO_FILE%" (
    echo ERROR: Audio file not found: %AUDIO_FILE%
    exit /b 1
)

if not exist "checkpoints\best_model.pt" (
    echo ERROR: Model checkpoint not found at checkpoints\best_model.pt
    echo Please train the model first: windows\train.bat
    exit /b 1
)

echo Config:     %CONFIG%
echo Audio File: %AUDIO_FILE%
echo.
echo Running inference...
echo ------------------------------------------

python scripts\predict.py --audio %AUDIO_FILE% --config %CONFIG%

echo ------------------------------------------
echo   Inference complete!
echo ==========================================
pause
