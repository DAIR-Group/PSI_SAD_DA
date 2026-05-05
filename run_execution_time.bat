@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "FEATURE_DIM=32"
set "DELTA=2"
set "N_SEEDS=500"
set "ANOMALY_RATE=0.0"
set "KNOWN_LABEL_RATE=0.1"
set "DEBUG_SEED=%~4"

if not "%DEBUG_SEED%"=="" (
  echo Debug mode enabled: running only seed=%DEBUG_SEED%
)

for %%N in (150) do (
  set "MODEL_NAME=deepsad_delta_%DELTA%_%%N_d%FEATURE_DIM%"

  if exist "%SCRIPT_DIR%models\!MODEL_NAME!_model.pth" (
    echo Model !MODEL_NAME! already exists, skipping training.
  ) else (
    echo Training model for n=%%N
    uv run python "%SCRIPT_DIR%train.py" --name !MODEL_NAME! --delta %DELTA% --n %%N --d %FEATURE_DIM% --anomaly-rate 0.05 --known-label-rate 0.1
    if errorlevel 1 (
      echo Training failed for n=%%N
      exit /b 1
    )
  )

  for %%V in (cuda) do (
    echo Running run_si_experiment.py for n=%%N on device=%%V
    if "%DEBUG_SEED%"=="" (
      uv run python "%SCRIPT_DIR%run_si_experiment.py" --delta %DELTA% --d %FEATURE_DIM% --anomaly-rate %ANOMALY_RATE% --known-label-rate %KNOWN_LABEL_RATE% --model-name !MODEL_NAME! --n %%N --n-seeds %N_SEEDS% --device %%V
    ) else (
      uv run python "%SCRIPT_DIR%run_si_experiment.py" --delta %DELTA% --d %FEATURE_DIM% --anomaly-rate %ANOMALY_RATE% --known-label-rate %KNOWN_LABEL_RATE% --model-name !MODEL_NAME! --n %%N --seed %DEBUG_SEED% --device %%V
    )
    if errorlevel 1 (
      echo Experiment failed for n=%%N on device=%%V
      exit /b 1
    )
  )
)

echo All execution-time runs completed.
endlocal
