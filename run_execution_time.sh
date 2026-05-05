#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_DIM=32
DELTA=2
N_SEEDS=120
ANOMALY_RATE=0.0
KNOWN_LABEL_RATE=0.1
DEBUG_SEED="${1:-}"

if [[ -n "${DEBUG_SEED}" ]]; then
  echo "Debug mode enabled: running only seed=${DEBUG_SEED}"
fi

for N in 150; do
  MODEL_NAME="deepsad_delta_${DELTA}_${N}_d${FEATURE_DIM}"

  if [[ -f "${SCRIPT_DIR}/models/${MODEL_NAME}_model.pth" ]]; then
    echo "Model ${MODEL_NAME} already exists, skipping training."
  else
    echo "Training model for n=${N}"
    if ! uv run python "${SCRIPT_DIR}/train.py" --name "${MODEL_NAME}" --delta "${DELTA}" --n "${N}" --d "${FEATURE_DIM}" --anomaly-rate 0.05 --known-label-rate 0.1; then
      echo "Training failed for n=${N}"
      exit 1
    fi
  fi

  for V in cpu cuda; do
    echo "Running run_si_experiment.py for n=${N} on device=${V}"
    if [[ -n "${DEBUG_SEED}" ]]; then
      RUN_ARGS=(--seed "${DEBUG_SEED}")
    else
      RUN_ARGS=(--n-seeds "${N_SEEDS}")
    fi

    if ! uv run python "${SCRIPT_DIR}/run_si_experiment.py" --delta "${DELTA}" --d "${FEATURE_DIM}" --anomaly-rate "${ANOMALY_RATE}" --known-label-rate "${KNOWN_LABEL_RATE}" --model-name "${MODEL_NAME}" --n "${N}" "${RUN_ARGS[@]}" --device "${V}"; then
      echo "Experiment failed for n=${N} on device=${V}"
      exit 1
    fi
  done
done

echo "All execution-time runs completed."
