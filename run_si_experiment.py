import argparse
import time
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kstest

from si import load_models


def main():
    parser = argparse.ArgumentParser(
        description="Run SI experiment with configurable parameters"
    )
    parser.add_argument(
        "--delta", type=float, default=0.0, help="Delta parameter for data generation"
    )
    parser.add_argument("--n", type=int, default=150, help="Number of instances")
    parser.add_argument("--d", type=int, default=10, help="Feature dimension")
    parser.add_argument(
        "--mu", type=float, default=0.0, help="Mu parameter for data generation"
    )
    parser.add_argument(
        "--target-mu",
        type=float,
        default=None,
        help="Target-domain synthetic mean. Defaults to --mu.",
    )
    parser.add_argument(
        "--source-mu",
        type=float,
        default=None,
        help="Source-domain synthetic mean. Defaults to --target-mu.",
    )
    parser.add_argument("--anomaly-rate", type=float, default=0.0, help="Anomaly rate")
    parser.add_argument(
        "--known-label-rate",
        type=float,
        default=0.1,
        help="Kept for compatibility with execution-time scripts; synthetic SI evaluation does not use labels.",
    )
    parser.add_argument(
        "--top-k-percent",
        type=float,
        default=0.1,
        help="Top k percent for anomaly detection",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=120, help="Number of seeds to run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to run; if omitted, runs seed range [0, n-seeds)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--covariance-dir",
        type=str,
        default="covariances",
        help="Directory containing saved covariance matrices",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepsad",
        help="Model name prefix (e.g. 'deepsad' loads deepsad_model.pth and deepsad_c.pth)",
    )
    parser.add_argument(
        "--h-dims",
        type=str,
        default="128,64,32",
        help="Hidden dims for the model, comma-separated",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=8,
        help="Representation dimension for the model",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "dnn_para"],
        help="Execution device for model and SI-DNN path",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="normal",
        choices=["normal", "oc", "bonferonni", "bonferroni", "naive", "no-inference"],
        help="Experiment method: normal (si.run), oc (si.run_oc), bonferonni (si.run_bonfer)",
    )
    parser.add_argument(
        "--top-k-normal-percent",
        type=float,
        default=0.3,
        help="Top k percent for selecting normal instances",
    )
    parser.add_argument(
        "--test-index-class",
        type=str,
        default="normal",
        choices=["normal", "anomaly"],
        help="Class of selected test point j.",
    )
    parser.add_argument(
        "--target-p-values",
        type=int,
        default=None,
        help="If set and --seed is omitted, keep increasing seeds from 0 until this many accepted p-values are collected.",
    )
    parser.add_argument(
        "--multiple-testing",
        type=bool,
        default=False,
        help="Find p-values for all test points in top k percent",
    )
    parser.add_argument(
        "--selection-event",
        type=str,
        default="j-in-o",
        choices=["j-in-o", "o-equal-oobs"],
        help="Selection event for the normal/proposed SI method.",
    )
    args = parser.parse_args()

    if args.method == "normal" and args.multiple_testing == False: 
        from si.run import run_one as run_fn
    elif args.method == "normal" and args.multiple_testing == True:
        from si.run import run_all as run_fn
    elif args.method == "oc":
        from si.run_oc import run as run_fn
    elif args.method in {"bonferonni", "bonferroni"}:
        from si.run_bonfer import run as run_fn
    elif args.method == "naive":
        from si.run_naive import run as run_fn
    elif args.method == "no-inference":
        from si.run_no_inference import run as run_fn

    os.makedirs(args.results_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "dnn_para":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    h_dims = [int(x.strip()) for x in args.h_dims.split(",")]
    model_input_dim = args.d
    target_mu = float(args.mu if args.target_mu is None else args.target_mu)
    source_mu = float(target_mu if args.source_mu is None else args.source_mu)

    deepsad_encoder, deepsad_c, _ = load_models(
        device=device,
        model_dir=args.model_dir,
        model_name=args.model_name,
        d=model_input_dim,
        h_dims=h_dims,
        rep_dim=args.rep_dim,
    )

    covariance_path = os.path.join(args.covariance_dir, f"{args.model_name}_cov.npy")
    Sigma = None
    if os.path.exists(covariance_path):
        Sigma = np.load(covariance_path)
        expected_shape = (model_input_dim, model_input_dim)
        if Sigma.shape != expected_shape:
            raise ValueError(
                f"Covariance shape mismatch at {covariance_path}: "
                f"expected {expected_shape}, got {Sigma.shape}"
            )
    else:
        print(f"Warning: covariance file not found at {covariance_path}. Falling back to identity.")

    p_values = []
    times = []

    def normalize_method_output(output) -> list[float]:
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

    if args.seed is None:
        if args.target_p_values is None:
            seeds = range(args.n_seeds)
            seed_mode = "range"
        else:
            if args.target_p_values <= 0:
                raise ValueError("--target-p-values must be > 0")
            seeds = None
            seed_mode = "until-target"
    else:
        if args.seed < 0:
            raise ValueError("--seed must be >= 0")
        seeds = [args.seed]
        seed_mode = "single"

    FPR = 0

    def build_runner_kwargs(seed: int) -> dict:
        kwargs = {
            "seed": seed,
            "delta": args.delta,
            "n": args.n,
            "d": args.d,
            "anomaly_rate": args.anomaly_rate,
            "top_k_percent": args.top_k_percent,
            "deepsad_encoder": deepsad_encoder,
            "deepsad_c": deepsad_c,
            "device": args.device,
            "test_index_class": args.test_index_class,
            "Sigma": Sigma,
        }
        if args.method == "normal" and args.multiple_testing:
            kwargs["mu"] = target_mu
        else:
            kwargs["target_mu"] = target_mu
            kwargs["source_mu"] = source_mu
        if args.method == "normal":
            kwargs["selection_event"] = args.selection_event
        return kwargs

    if seed_mode == "until-target":
        seed = 0
        while len(p_values) < args.target_p_values:
            start = time.time()
            list_p_value = normalize_method_output(run_fn(**build_runner_kwargs(seed)))
            seed += 1
            if len(list_p_value) == 0:
                continue
            p_values.extend(list_p_value)
            for p_value in list_p_value:
                if p_value < 0.05:
                    FPR += 1
            times.append(time.time() - start)
    else:
        for seed in seeds:
            start = time.time()
            list_p_value = normalize_method_output(run_fn(**build_runner_kwargs(seed)))
            if len(list_p_value) == 0:
                continue
            p_values.extend(list_p_value)
            for p_value in list_p_value:
                if p_value < 0.05:
                    FPR += 1
            times.append(time.time() - start)

    print(f"Collected {len(p_values)} p-values with FPs: {FPR}")
    if len(p_values) == 0:
        raise RuntimeError("No valid p-values were produced.")

    suffix = f"delta_{args.delta}_n_{args.n}"
    dir_name = f"{args.model_name}_{suffix}_method_{args.method}_{args.test_index_class}"
    print(f"Results for {dir_name}:")
    # Create result dir if it doesn't exist
    os.makedirs(os.path.join(args.results_dir, dir_name), exist_ok=True)

    print(f"Total time taken: {sum(times)} seconds")
    print(f"Avg time per run: {sum(times) / len(p_values)} seconds")
    with open(os.path.join(args.results_dir, dir_name, f"timing.txt"), "w") as f:
        for t in times:
            f.write(f"{t}\n")

    # Save results
    result_prefix = suffix

    with open(os.path.join(args.results_dir, dir_name, f"p_values.txt"), "w") as f:
        for p in p_values:
            f.write(f"{p}\n")

    # np.save(
    #     os.path.join(args.results_dir, dir_name, f"{result_prefix}_p_values.npy"),
    #     np.array(p_values),
    # )

    # Plot histogram
    plt.hist(p_values)
    plt.title(f"Histogram of p-values from SI (delta={args.delta}, n={args.n})")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.results_dir, dir_name, f"histogram.png"))
    plt.close()

    # KS test
    ks_result = kstest(p_values, "uniform")
    print(f"KS test against uniform distribution: {ks_result}")

    with open(os.path.join(args.results_dir, dir_name, f"ks_test.txt"), "w") as f:
        f.write(f"KS test against uniform distribution: {ks_result}\n")


if __name__ == "__main__":
    main()
