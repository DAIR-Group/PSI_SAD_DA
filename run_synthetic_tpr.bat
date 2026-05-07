@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "RESULTS_ROOT=%SCRIPT_DIR%results\synthetic_tpr"
set "INDEPENDENT_RESULTS_DIR=%RESULTS_ROOT%\independent_data"
set "CORRELATED_RESULTS_DIR=%RESULTS_ROOT%\correlated_data"

set "ENABLE_DA=1"
set "MODEL_PREFIX=deepsad"
set "TARGET_MU=0.0"
set "SOURCE_MU=1.0"
set "TRAIN_DA_ARGS="
if defined ENABLE_DA (
  set "MODEL_PREFIX=deepsad_da_delta2"
  set "TRAIN_DA_ARGS=--enable-da --da-generator-hidden-dims 64,32,10 --da-critic-hidden-dims 64,32,16 --da-epochs 50 --da-critic-steps 5 --source-mu %SOURCE_MU%"
)

set "INDEPENDENT_MODEL_NAME=%MODEL_PREFIX%_independent"
set "CORRELATED_MODEL_NAME=%MODEL_PREFIX%_correlated"

set "D=10"
set "TRAIN_N=6000"
set "REP_DIM=8"
set "H_DIMS=64,32"
set "AE_EPOCHS=30"
set "SAD_EPOCHS=150"
set "LR=0.0005"
set "BATCH_SIZE=128"
set "TRAIN_DELTA=2.0"
set "TRAIN_ANOMALY_RATE=0.05"
set "KNOWN_LABEL_RATE=0.1"
set "N_SEEDS=500"
set "SOURCE_TEST_SIZE=200"
set "TARGET_TEST_SIZE=100"
set "N_FIXED=300"
set "DELTA_LIST=0.5 1.0 1.5 2.0"
set "DELTA_CSV=0.5,1.0,1.5,2.0"
set "ANOMALY_RATE=0.05"
set "REFERENCE_SIZE=200"
set "ALPHA=0.05"
set "METHODS=proposed,oc,bonferroni,naive"
set "TEST_INDEX_CLASS=anomaly"
set "METRIC_NAME=tpr"

if not exist "%RESULTS_ROOT%" mkdir "%RESULTS_ROOT%"
if not exist "%INDEPENDENT_RESULTS_DIR%" mkdir "%INDEPENDENT_RESULTS_DIR%"
if not exist "%CORRELATED_RESULTS_DIR%" mkdir "%CORRELATED_RESULTS_DIR%"

call :ensure_model 0.0 "%INDEPENDENT_MODEL_NAME%"
if errorlevel 1 exit /b 1

call :ensure_model 0.5 "%CORRELATED_MODEL_NAME%"
if errorlevel 1 exit /b 1

call :run_experiment independent 0.0 "%INDEPENDENT_MODEL_NAME%" "%INDEPENDENT_RESULTS_DIR%"
if errorlevel 1 exit /b 1

call :run_experiment correlated 0.5 "%CORRELATED_MODEL_NAME%" "%CORRELATED_RESULTS_DIR%"
if errorlevel 1 exit /b 1

echo All synthetic TPR experiments completed.
echo Independent figure: "%INDEPENDENT_RESULTS_DIR%\final_tpr_plot.pdf"
echo Correlated figure: "%CORRELATED_RESULTS_DIR%\final_tpr_plot.pdf"
endlocal
exit /b 0

:ensure_model
set "RHO=%~1"
set "MODEL_NAME=%~2"
set "MODEL_READY=0"
if exist "%SCRIPT_DIR%models\%MODEL_NAME%_model.pth" if exist "%SCRIPT_DIR%models\%MODEL_NAME%_c.pth" if exist "%SCRIPT_DIR%covariances\%MODEL_NAME%_cov.npy" if exist "%SCRIPT_DIR%models\%MODEL_NAME%_metadata.json" (
  if defined ENABLE_DA (
    if exist "%SCRIPT_DIR%models\%MODEL_NAME%_source.npy" if exist "%SCRIPT_DIR%covariances\%MODEL_NAME%_source_cov.npy" (
      findstr /C:"source_target_train_only_v1" "%SCRIPT_DIR%models\%MODEL_NAME%_metadata.json" >nul
      if not errorlevel 1 set "MODEL_READY=1"
    )
  ) else (
    findstr /C:"source_target_train_only_v1" "%SCRIPT_DIR%models\%MODEL_NAME%_metadata.json" >nul
    if not errorlevel 1 set "MODEL_READY=1"
  )
)
if "%MODEL_READY%"=="1" (
  echo Using existing synthetic model artifacts for %MODEL_NAME%
  exit /b 0
)

echo Training synthetic model %MODEL_NAME% with rho=%RHO%
python "%SCRIPT_DIR%train.py" ^
  --name %MODEL_NAME% ^
  --d %D% ^
  --n %TRAIN_N% ^
  --target-mu %TARGET_MU% ^
  --delta %TRAIN_DELTA% ^
  --anomaly-rate %TRAIN_ANOMALY_RATE% ^
  --known-label-rate %KNOWN_LABEL_RATE% ^
  --rho %RHO% ^
  --n-reference %REFERENCE_SIZE% ^
  --h-dims %H_DIMS% ^
  --rep-dim %REP_DIM% ^
  --ae-epochs %AE_EPOCHS% ^
  --sad-epochs %SAD_EPOCHS% ^
  --lr %LR% ^
  --batch-size %BATCH_SIZE% ^
  --model-dir "%SCRIPT_DIR%models" ^
  --covariance-dir "%SCRIPT_DIR%covariances" ^
  %TRAIN_DA_ARGS%

if errorlevel 1 (
  echo Synthetic model training failed for %MODEL_NAME%
  exit /b 1
)
exit /b 0

:run_experiment
set "EXPERIMENT_NAME=%~1"
set "RHO=%~2"
set "MODEL_NAME=%~3"
set "RESULTS_DIR=%~4"

echo ==========================================
echo Running %EXPERIMENT_NAME% synthetic TPR experiment
echo rho=%RHO%, results="%RESULTS_DIR%"
echo source_test_size=%SOURCE_TEST_SIZE%, target_test_size=%TARGET_TEST_SIZE%, total n=%N_FIXED%, delta_list=%DELTA_LIST%
echo ==========================================

for %%D in (%DELTA_LIST%) do (
  python "%SCRIPT_DIR%scripts\run_synthetic_experiment.py" ^
    --rho %RHO% ^
    --target-mu %TARGET_MU% ^
    --source-mu %SOURCE_MU% ^
    --target-rho %RHO% ^
    --source-rho %RHO% ^
    --delta %%D ^
    --d %D% ^
    --n %N_FIXED% ^
    --source-test-size %SOURCE_TEST_SIZE% ^
    --target-test-size %TARGET_TEST_SIZE% ^
    --n-seeds %N_SEEDS% ^
    --model-name %MODEL_NAME% ^
    --model-dir "%SCRIPT_DIR%models" ^
    --covariance-dir "%SCRIPT_DIR%covariances" ^
    --h-dims %H_DIMS% ^
    --rep-dim %REP_DIM% ^
    --anomaly-rate %ANOMALY_RATE% ^
    --test-index-class %TEST_INDEX_CLASS% ^
    --results-dir "%RESULTS_DIR%" ^
    --methods %METHODS% ^
    --include-no-inference ^
    --alpha %ALPHA% ^
    --reference-size %REFERENCE_SIZE%

  if errorlevel 1 (
    echo Synthetic TPR run failed at delta=%%D for %EXPERIMENT_NAME%
    exit /b 1
  )
)

python "%SCRIPT_DIR%scripts\plot_synthetic_rate.py" ^
  --results-dir "%RESULTS_DIR%" ^
  --x-axis delta ^
  --n %N_FIXED% ^
  --delta-list %DELTA_CSV% ^
  --x-label "Delta" ^
  --alpha %ALPHA% ^
  --metric-name %METRIC_NAME% ^
  --methods proposed,oc,bonferroni ^
  --output "%RESULTS_DIR%\final_tpr_plot.pdf"

if errorlevel 1 (
  echo Synthetic TPR plotting failed for %EXPERIMENT_NAME%.
  exit /b 1
)

python "%SCRIPT_DIR%scripts\plot_synthetic_rate.py" ^
  --results-dir "%RESULTS_DIR%" ^
  --x-axis delta ^
  --n %N_FIXED% ^
  --delta-list %DELTA_CSV% ^
  --x-label "Delta" ^
  --alpha %ALPHA% ^
  --metric-name %METRIC_NAME% ^
  --methods proposed,oc,bonferroni ^
  --output "%RESULTS_DIR%\final_tpr_plot.png"

if errorlevel 1 (
  echo Synthetic TPR PNG plotting failed for %EXPERIMENT_NAME%.
  exit /b 1
)

exit /b 0
