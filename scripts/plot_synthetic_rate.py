from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    style="white",
    context="paper",
    font="serif",
)

DISPLAY_NAMES = {
    "proposed": "PSI-SAD-DA",
    "oc": "OC",
    "bonferroni": "Bonferroni",
    "naive": "Naive",
    "no_inference": "No-Inference",
    "wo_ad": "w/o-SAD",
    "wo_da": "w/o-DA",
}

COLORS = [
    "#d94b6a",
    "#f08a75",
    "#7fc8ae",
    "#6b88d9",
    "#c2be7a",
    "#F360D3",
]

METHOD_COLORS = {
    "proposed": COLORS[0],
    "oc": COLORS[3],
    "bonferroni": COLORS[4],
    "naive": COLORS[5],
    "wo_ad": COLORS[1],
    "wo_da": COLORS[2],
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


def load_method_metric(results_dir: str, delta: float, n: int, metric_name: str, method: str) -> float:
    summary = load_summary(results_dir, delta, n, metric_name)
    if method in summary.get("methods", {}):
        return float(summary["methods"][method][metric_name])

    debug_summary = os.path.join(
        results_dir,
        f"delta_{delta}_n_{n}",
        "debug",
        method,
        "summary.json",
    )
    if not os.path.exists(debug_summary):
        raise KeyError(
            f"Method '{method}' is missing from {summary_path(results_dir, delta, n, metric_name)} "
            f"and debug summary {debug_summary} does not exist."
        )
    with open(debug_summary, "r", encoding="utf-8") as handle:
        method_summary = json.load(handle)
    return float(method_summary[metric_name])


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
    parser.add_argument("--methods", type=str, default="proposed,wo_ad,wo_da,oc,bonferroni,naive")
    parser.add_argument("--x-label", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Show the plot window after saving.")
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    curves: Dict[str, List[float]] = {method: [] for method in methods}

    if args.x_axis == "n":
        if args.delta is None or args.n_list is None:
            raise ValueError("--x-axis=n requires both --delta and --n-list.")
        x_values = parse_int_list(args.n_list)
        x_label = args.x_label or "Number of source test samples"
        for n_value in x_values:
            for method in methods:
                curves[method].append(
                    load_method_metric(args.results_dir, args.delta, n_value, args.metric_name, method)
                )
    else:
        if args.n is None or args.delta_list is None:
            raise ValueError("--x-axis=delta requires both --n and --delta-list.")
        x_values = parse_float_list(args.delta_list)
        x_label = args.x_label or "Delta"
        for delta_value in x_values:
            for method in methods:
                curves[method].append(
                    load_method_metric(args.results_dir, delta_value, args.n, args.metric_name, method)
                )

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for idx, method in enumerate(methods):
        ax.plot(
            x_values,
            curves[method],
            marker="o",
            linewidth=1.5,
            color=METHOD_COLORS.get(method, COLORS[idx % len(COLORS)]),
            label=DISPLAY_NAMES.get(method, method),
        )

    ax.set_xlabel(x_label, fontweight="bold", fontsize=14)
    ax.set_ylabel(str(args.metric_name).upper(), fontweight="bold", fontsize=14)
    ax.set_xticks(x_values)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=14)

    ax.grid(False)
    ax.legend(frameon=True, fontsize=12)
    plt.tight_layout()

    if args.x_axis == "n":
        default_name = f"delta_{args.delta}_{args.metric_name}_plot.pdf"
    else:
        default_name = f"n_{args.n}_{args.metric_name}_vs_delta_plot.pdf"
    output_path = args.output or os.path.join(args.results_dir, default_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()

