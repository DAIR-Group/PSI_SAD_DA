from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scripts.result_schema import method_summary, parse_methods, save_method_outputs
from si.util import (
    load_models,
    load_working_model_artifacts,
    resolve_source_target_test_sizes,
)

proposed_run = importlib.import_module("si.run")
run_oc = importlib.import_module("si.run_oc")
run_bonfer = importlib.import_module("si.run_bonfer")
run_naive = importlib.import_module("si.run_naive")
run_no_inference = importlib.import_module("si.run_no_inference")
run_wo_ad = importlib.import_module("si.wo_AD.run")
run_wo_da = importlib.import_module("si.wo_DA.run")

METHOD_RUNNERS = {
    "proposed": proposed_run.run_one,
    "wo_ad": run_wo_ad.run,
    "wo_da": run_wo_da.run,
    "oc": run_oc.run,
    "bonferroni": run_bonfer.run,
    "naive": run_naive.run,
    "no_inference": run_no_inference.run,
}


def resolve_device(requested_device: str) -> str:
    requested_device = str(requested_device).lower()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "dnn_para":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda, dnn_para")
    return requested_device


def device_log_message(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        try:
            gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            gpu_name = "unknown"
        return f"execution_device=cuda gpu='{gpu_name}'"
    return "execution_device=cpu"


def default_model_name(rho: float) -> str:
    return "deepsad_independent" if abs(float(rho)) < 1e-12 else "deepsad_correlated"


def make_default_synthetic_covariance(d: int, rho: float) -> np.ndarray:
    if abs(float(rho)) < 1e-12:
        return np.eye(int(d), dtype=float)
    idx = np.arange(int(d))
    return float(rho) ** np.abs(np.subtract.outer(idx, idx))


def load_working_artifacts(
    model_dir: str,
    model_name: str,
    d: int,
    rho: float,
    allow_fallback: bool,
) -> tuple[Dict, Dict]:
    expected_shape = (int(d), int(d))
    artifacts = load_working_model_artifacts(
        model_dir=model_dir,
        model_name=model_name,
        allow_missing_source=bool(allow_fallback),
    )

    Sigma_target = artifacts["Sigma_target"]
    target_fallback_used = False
    if Sigma_target is not None:
        if Sigma_target.shape != expected_shape:
            raise ValueError(
                f"Covariance shape mismatch for model '{model_name}': "
                f"expected {expected_shape}, got {Sigma_target.shape}."
            )
    else:
        if not allow_fallback:
            raise FileNotFoundError(
                f"Missing target covariance artifact for model '{model_name}'. "
                "Train the synthetic model first or pass --allow-artifact-fallback."
            )
        Sigma_target = make_default_synthetic_covariance(d, rho)
        target_fallback_used = True

    Sigma_source = artifacts["Sigma_source"]
    source_fallback_used = False
    if Sigma_source is None and len(artifacts["X_source"]) > 0:
        if not allow_fallback:
            raise FileNotFoundError(
                f"Missing source covariance artifact for model '{model_name}'. "
                "Retrain the DA model to produce exact source/target block covariance artifacts."
            )
        Sigma_source = np.asarray(Sigma_target, dtype=np.float64)
        source_fallback_used = True

    artifacts["Sigma_target"] = np.asarray(Sigma_target, dtype=np.float64)
    artifacts["Sigma_ref"] = np.asarray(Sigma_target, dtype=np.float64)
    artifacts["Sigma_source"] = None if Sigma_source is None else np.asarray(Sigma_source, dtype=np.float64)
    metadata = dict(artifacts.get("metadata", {}))
    return artifacts, {
        "source": "working_model_artifacts",
        "target_covariance_path": metadata.get("target_covariance_path") or metadata.get("covariance_path"),
        "source_covariance_path": metadata.get("source_covariance_path"),
        "source_data_path": metadata.get("source_data_path"),
        "fallback_used": bool(target_fallback_used or source_fallback_used),
        "target_fallback_used": bool(target_fallback_used),
        "source_fallback_used": bool(source_fallback_used),
        "has_source_block": bool(len(artifacts["X_source"]) > 0),
    }


def normalize_method_output(output) -> List[float]:
    if output is None:
        return []
    if isinstance(output, (float, int, np.floating, np.integer)):
        output = [float(output)]
    normalized = []
    for value in output:
        if value is None:
            continue
        normalized.append(float(value))
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Run synthetic SI experiments for fixed (delta, n, rho).")
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--target-mu", type=float, default=None)
    parser.add_argument("--source-mu", type=float, default=None)
    parser.add_argument("--target-rho", type=float, default=None)
    parser.add_argument("--source-rho", type=float, default=None)
    parser.add_argument("--anomaly-rate", type=float, default=0.0)
    parser.add_argument("--top-k-percent", type=float, default=0.05)
    parser.add_argument("--reference-size", type=int, default=200)
    parser.add_argument("--source-test-size", type=int, default=None)
    parser.add_argument("--target-test-size", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--covariance-dir", type=str, default="covariances")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--h-dims", type=str, default="64,32")
    parser.add_argument("--rep-dim", type=int, default=8)
    parser.add_argument("--results-dir", type=str, default="results/synthetic")
    parser.add_argument("--methods", type=str, default="proposed,oc,bonferroni,naive")
    parser.add_argument("--include-no-inference", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--test-index-class", type=str, default="normal", choices=["normal", "anomaly"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dnn_para"])
    parser.add_argument("--allow-artifact-fallback", action="store_true")
    parser.add_argument(
        "--selection-event",
        type=str,
        default="j-in-o",
        choices=["j-in-o", "o-equal-oobs"],
        help="Selection event for the proposed SI method.",
    )
    args = parser.parse_args()

    if int(args.reference_size) <= 0:
        raise ValueError("--reference-size must be positive.")
    if int(args.n_seeds) <= 0:
        raise ValueError("--n-seeds must be positive.")
    n_total_effective, n_source_effective, n_target_effective = resolve_source_target_test_sizes(
        args.n,
        source_test_size=args.source_test_size,
        target_test_size=args.target_test_size,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    resolved_device = resolve_device(args.device)
    print(device_log_message(resolved_device), flush=True)

    methods = parse_methods(args.methods)
    if args.include_no_inference and "no_inference" not in methods:
        methods.append("no_inference")
    unsupported = [method for method in methods if method not in METHOD_RUNNERS]
    if unsupported:
        raise ValueError(f"Unsupported methods requested: {unsupported}")

    model_name = args.model_name or default_model_name(args.rho)
    h_dims = [int(value.strip()) for value in args.h_dims.split(",") if value.strip()]
    encoder, deepsad_c, _ = load_models(
        device=resolved_device,
        model_dir=args.model_dir,
        model_name=model_name,
        d=args.d,
        h_dims=h_dims,
        rep_dim=args.rep_dim,
    )
    working_artifacts, covariance_debug = load_working_artifacts(
        model_dir=args.model_dir,
        model_name=model_name,
        d=args.d,
        rho=args.rho,
        allow_fallback=bool(args.allow_artifact_fallback),
    )
    Sigma_target = working_artifacts["Sigma_target"]
    Sigma_source = working_artifacts["Sigma_source"]
    X_source_obs = working_artifacts["X_source"]
    model_metadata = dict(working_artifacts.get("metadata", {}))
    target_mu = float(
        args.mu
        if args.target_mu is None
        else args.target_mu
    )
    if args.target_mu is None and "target_mu" in model_metadata:
        target_mu = float(model_metadata["target_mu"])
    source_mu = float(
        model_metadata.get("source_mu", target_mu + 2.0)
        if args.source_mu is None
        else args.source_mu
    )
    target_rho = float(
        args.rho
        if args.target_rho is None
        else args.target_rho
    )
    if args.target_rho is None and "target_rho" in model_metadata:
        target_rho = float(model_metadata["target_rho"])
    source_rho = float(
        model_metadata.get("source_rho", target_rho)
        if args.source_rho is None
        else args.source_rho
    )

    metric_name = "fpr" if str(args.test_index_class) == "normal" else "tpr"
    result_prefix = f"delta_{args.delta}_n_{args.n}"
    experiment_dir = os.path.join(args.results_dir, result_prefix)
    debug_dir = os.path.join(experiment_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    p_values_by_method: Dict[str, List[float]] = {method: [] for method in methods}
    n_attempted_by_method = {method: 0 for method in methods}
    per_method_runtime = {method: 0.0 for method in methods}
    run_start = time.time()

    for offset in range(int(args.n_seeds)):
        seed = int(args.global_seed + offset)
        if args.progress_every > 0 and (
            (offset + 1) == 1
            or (offset + 1) % int(args.progress_every) == 0
            or (offset + 1) == int(args.n_seeds)
        ):
            print(f"[synthetic] seed_attempt={offset + 1}/{args.n_seeds}", flush=True)

        for method in methods:
            n_attempted_by_method[method] += 1

            runner = METHOD_RUNNERS[method]
            method_start = time.time()
            runner_kwargs = {
                "seed": seed,
                "delta": args.delta,
                "n": args.n,
                "target_mu": target_mu,
                "source_mu": source_mu,
                "d": args.d,
                "anomaly_rate": args.anomaly_rate,
                "top_k_percent": args.top_k_percent,
                "deepsad_encoder": encoder,
                "deepsad_c": deepsad_c,
                "device": args.device,
                "Sigma": Sigma_target,
                "Sigma_target": Sigma_target,
                "Sigma_source": Sigma_source,
                "X_source_obs": X_source_obs,
                "target_rho": target_rho,
                "source_rho": source_rho,
                "reference_size": args.reference_size,
                "source_test_size": args.source_test_size,
                "target_test_size": args.target_test_size,
                "test_index_class": args.test_index_class,
            }
            if method == "proposed":
                runner_kwargs["selection_event"] = args.selection_event
            method_output = runner(**runner_kwargs)
            per_method_runtime[method] += time.time() - method_start
            p_values_by_method[method].extend(normalize_method_output(method_output))

    elapsed = time.time() - run_start

    for method in methods:
        save_method_outputs(
            debug_dir,
            method,
            p_values_by_method[method],
            n_attempted=n_attempted_by_method[method],
            alpha=args.alpha,
            metric_name=metric_name,
            rate_denominator="pvalues",
        )

    summary = {
        "delta": float(args.delta),
        "n": int(args.n),
        "n_total_test": int(n_total_effective),
        "n_source_test": int(n_source_effective),
        "n_target_test": int(n_target_effective),
        "source_test_size_arg": None if args.source_test_size is None else int(args.source_test_size),
        "target_test_size_arg": None if args.target_test_size is None else int(args.target_test_size),
        "rho": float(args.rho),
        "target_mu": float(target_mu),
        "source_mu": float(source_mu),
        "target_rho": float(target_rho),
        "source_rho": float(source_rho),
        "anomaly_rate": float(args.anomaly_rate),
        "reference_size": int(args.reference_size),
        "global_seed": int(args.global_seed),
        "n_seeds": int(args.n_seeds),
        "metric_name": metric_name,
        "test_index_class": str(args.test_index_class),
        "alpha": float(args.alpha),
        "top_k_percent": float(args.top_k_percent),
        "selection_event": str(args.selection_event),
        "device": str(resolved_device),
        "model_name": str(model_name),
        "model_dir": str(args.model_dir),
        "covariance_debug": dict(covariance_debug),
        "elapsed_seconds": float(elapsed),
        "debug_dir": debug_dir,
        "methods": {},
    }

    for method in methods:
        summary["methods"][method] = {
            **method_summary(
                p_values_by_method[method],
                n_attempted=n_attempted_by_method[method],
                alpha=args.alpha,
                metric_name=metric_name,
                rate_denominator="pvalues",
            ),
            "runtime_seconds": float(per_method_runtime[method]),
        }

    summary_path = os.path.join(args.results_dir, f"{result_prefix}_{metric_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(experiment_dir, f"{metric_name}_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(
        f"[synthetic] finished delta={args.delta} n={args.n} rho={args.rho} metric={metric_name} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

