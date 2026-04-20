@echo off
REM DepressoSpeech - Full Inference Pipeline (Windows)
REM Starts the API server and keeps it running

echo ==========================================
echo   DepressoSpeech - Inference Pipeline
echo ==========================================

cd /d "%~dp0\.."
call .venv\Scripts\activate.bat

if "%CONFIG%"=="" set CONFIG=configs\inference_config.yaml
if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=8000

REM Check all artifacts
set MISSING=0
if not exist "checkpoints\best_model.pt" (
    echo ERROR: Missing checkpoints\best_model.pt
    set MISSING=1
)
if not exist "checkpoints\scalers\feature_scalers.pkl" (
    echo ERROR: Missing checkpoints\scalers\feature_scalers.pkl
    set MISSING=1
)
if not exist "checkpoints\scalers\pca_reducer.pkl" (
    echo ERROR: Missing checkpoints\scalers\pca_reducer.pkl
    set MISSING=1
)
if %MISSING%==1 (
    echo.
    echo Run the training pipeline first: windows\run_training_pipeline.bat
    exit /b 1
)

echo All model artifacts found.
echo.
echo Starting API server on %HOST%:%PORT% ...
echo Server will keep running. Use Ctrl+C to stop.
echo.
echo Test with:
echo   curl http://%HOST%:%PORT%/health
echo   curl -X POST -F "file=@audio.wav" http://%HOST%:%PORT%/predict
echo ------------------------------------------

python scripts\serve.py --host %HOST% --port %PORT% --config %CONFIG% --log-level info
