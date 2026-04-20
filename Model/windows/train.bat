@echo off
REM DepressoSpeech - Training Pipeline (Windows)
REM Trains the DepressionModel (MLP + BiGRU + Attention)

echo ==========================================
echo   DepressoSpeech - Training Pipeline
echo ==========================================

cd /d "%~dp0\.."
call .venv\Scripts\activate.bat

if "%CONFIG%"=="" set CONFIG=configs\training_config.yaml
if "%FEATURE_DIR%"=="" set FEATURE_DIR=data\features

echo Training Configuration:
echo   Config:      %CONFIG%
echo   Feature Dir: %FEATURE_DIR%
echo.

REM Check GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Device: CPU')"

REM Ensure checkpoint dirs
if not exist "checkpoints\scalers" mkdir checkpoints\scalers

echo.
echo Starting training...
echo ------------------------------------------

if "%USE_PRECOMPUTED_PCA%"=="" (
    python scripts\train.py --config %CONFIG% --feature-dir %FEATURE_DIR%
) else (
    python scripts\train.py --config %CONFIG% --feature-dir %FEATURE_DIR% --use-precomputed-pca
)

echo ------------------------------------------
echo   Training complete!
echo   Artifacts:
echo     Model:   checkpoints\best_model.pt
echo     Scalers: checkpoints\scalers\feature_scalers.pkl
echo     PCA:     checkpoints\scalers\pca_reducer.pkl
echo ==========================================
pause
