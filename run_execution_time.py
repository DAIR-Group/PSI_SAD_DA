#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], script_dir: Path) -> None:
    result = subprocess.run(command, cwd=script_dir)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def is_writable_dir(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    return os.access(path.parent, os.W_OK)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Python version of run_execution_time.bat (Kaggle-friendly)."
    )
    parser.add_argument(
        "--debug-seed",
        type=int,
        default=None,
        help="If set, run only this seed instead of --n-seeds.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save experiment outputs.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    kaggle_working = Path("/kaggle/working")

    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "models"
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else (
            kaggle_working / "results"
            if kaggle_working.exists()
            else script_dir / "results"
        )
    )

    train_model_dir = model_dir
    if not is_writable_dir(train_model_dir) and kaggle_working.exists():
        train_model_dir = kaggle_working / "models"

    results_dir.mkdir(parents=True, exist_ok=True)

    feature_dim = 32
    delta = 2
    n_seeds = 30
    anomaly_rate = 0.0
    known_label_rate = 0.1

    if args.debug_seed is not None:
        print(f"Debug mode enabled: running only seed={args.debug_seed}")

    for n in (2500, 5000, 7500, 10000):
        model_name = f"deepsad_delta_{delta}_{n}_d{feature_dim}"
        model_path = model_dir / f"{model_name}_model.pth"
        model_source_dir = model_dir

        if model_path.exists():
            print(f"Model {model_name} already exists, skipping training.")
        else:
            print(f"Training model for n={n}")
            model_source_dir = train_model_dir
            train_cmd = [
                sys.executable,
                str(script_dir / "train.py"),
                "--name",
                model_name,
                "--delta",
                str(delta),
                "--n",
                str(n),
                "--d",
                str(feature_dim),
                "--anomaly-rate",
                "0.05",
                "--known-label-rate",
                "0.1",
                "--model-dir",
                str(model_source_dir),
            ]

            try:
                run_command(train_cmd, script_dir)
            except subprocess.CalledProcessError:
                print(f"Training failed for n={n}")
                return 1

        for device in ("cuda",):
            print(f"Running run_si_experiment.py for n={n} on device={device}")
            run_cmd = [
                sys.executable,
                str(script_dir / "run_si_experiment.py"),
                "--delta",
                str(delta),
                "--d",
                str(feature_dim),
                "--anomaly-rate",
                str(anomaly_rate),
                "--known-label-rate",
                str(known_label_rate),
                "--model-name",
                model_name,
                "--n",
                str(n),
                "--model-dir",
                str(model_source_dir),
                "--results-dir",
                str(results_dir),
                "--device",
                device,
            ]

            if args.debug_seed is None:
                run_cmd.extend(["--n-seeds", str(n_seeds)])
            else:
                run_cmd.extend(["--seed", str(args.debug_seed)])

            try:
                run_command(run_cmd, script_dir)
            except subprocess.CalledProcessError:
                print(f"Experiment failed for n={n} on device={device}")
                return 1

    print("All execution-time runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
