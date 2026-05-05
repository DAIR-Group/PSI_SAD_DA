from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

DISPLAY_NAMES = {
    "proposed": "PSI-SAD-DA",
    "oc": "PSI-SAD-DA-oc",
    "bonferroni": "Bonferroni",
    "naive": "Naive",
    "no_inference": "No-Inference",
}


def parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in str(raw).split(",") if str(x).strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in str(raw).split(",") if str(x).strip()]


def summary_path(results_dir: str, delta: float, n: int, metric_name: str) -> str:
    return os.path.join(results_dir, f"delta_{delta}_n_{n}_{metric_name}_summary.json")


def load_summary(results_dir: str, delta: float, n: int, metric_name: str) -> Dict:
    path = summary_path(results_dir, delta, n, metric_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing synthetic summary file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Plot synthetic FPR/TPR curves across n or delta.")
    parser.add_argument("--results-dir", type=str, default="results/synthetic")
    parser.add_argument("--x-axis", type=str, default="n", choices=["n", "delta"])
    parser.add_argument("--delta", type=float, default=None, help="Fixed delta when plotting across n.")
    parser.add_argument("--n", type=int, default=None, help="Fixed n when plotting across delta.")
    parser.add_argument("--n-list", type=str, default=None, help="Comma-separated n list when --x-axis=n.")
    parser.add_argument("--delta-list", type=str, default=None, help="Comma-separated delta list when --x-axis=delta.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--metric-name", type=str, required=True, choices=["fpr", "tpr"])
    parser.add_argument("--methods", type=str, default="proposed,oc,bonferroni,naive,no_inference")
    parser.add_argument("--x-label", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    curves: Dict[str, List[float]] = {method: [] for method in methods}

    if args.x_axis == "n":
        if args.delta is None or args.n_list is None:
            raise ValueError("--x-axis=n requires both --delta and --n-list.")
        x_values = parse_int_list(args.n_list)
        x_label = args.x_label or "Dataset Size (n)"
        for n_value in x_values:
            summary = load_summary(args.results_dir, args.delta, n_value, args.metric_name)
            for method in methods:
                value = summary["methods"][method][args.metric_name]
                curves[method].append(float(value))
    else:
        if args.n is None or args.delta_list is None:
            raise ValueError("--x-axis=delta requires both --n and --delta-list.")
        x_values = parse_float_list(args.delta_list)
        x_label = args.x_label or "Signal Strength (delta)"
        for delta_value in x_values:
            summary = load_summary(args.results_dir, delta_value, args.n, args.metric_name)
            for method in methods:
                value = summary["methods"][method][args.metric_name]
                curves[method].append(float(value))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method in methods:
        ax.plot(
            x_values,
            curves[method],
            marker="o",
            linewidth=1.5,
            label=DISPLAY_NAMES.get(method, method),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(str(args.metric_name).upper())
    ax.set_xticks(x_values)
    ax.set_ylim(-0.05, 1.05)
    if str(args.metric_name).lower() == "fpr":
        ax.axhline(
            float(args.alpha),
            color="#ff8fb3",
            linestyle="--",
            linewidth=1.0,
            label=f"alpha={args.alpha:g}",
        )

    ax.grid(False)
    ax.legend()
    fig.tight_layout()

    if args.x_axis == "n":
        default_name = f"delta_{args.delta}_{args.metric_name}_plot.pdf"
    else:
        default_name = f"n_{args.n}_{args.metric_name}_vs_delta_plot.pdf"
    output_path = args.output or os.path.join(args.results_dir, default_name)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()

