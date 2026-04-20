@echo off
REM DepressoSpeech - Full Training Pipeline (Windows)
REM Runs: setup -> extract features -> train

echo ==========================================
echo   DepressoSpeech - Full Training Pipeline
echo ==========================================
echo.

cd /d "%~dp0\.."

echo [Step 1/3] Setting up environment...
call windows\setup.bat

echo.
echo [Step 2/3] Extracting features...
call windows\extract_features.bat

echo.
echo [Step 3/3] Training model...
call windows\train.bat

echo.
echo ==========================================
echo   Full training pipeline complete!
echo.
echo   Next steps:
echo     Single prediction:  windows\predict.bat ^<audio.wav^>
echo     Batch prediction:   windows\predict_batch.bat ^<audio_dir\^>
echo     Start API server:   windows\serve.bat
echo ==========================================
pause
