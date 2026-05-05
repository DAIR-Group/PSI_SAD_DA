from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest


METHOD_ALIASES = {
    "proposed": "proposed",
    "normal": "proposed",
    "oc": "oc",
    "overconditioning": "oc",
    "over_conditioning": "oc",
    "bonferroni": "bonferroni",
    "bonferonni": "bonferroni",
    "naive": "naive",
    "no_inference": "no_inference",
    "no-inference": "no_inference",
}

DISPLAY_NAMES = {
    "proposed": "Proposed",
    "oc": "Over Conditioning",
    "bonferroni": "Bonferroni",
    "naive": "Naive",
    "no_inference": "No-Inference",
}

SUPPORTED_METHODS = ("proposed", "oc", "bonferroni", "naive", "no_inference")


def canonicalize_method_name(method_name: str) -> str:
    normalized = str(method_name).strip().lower().replace("-", "_")
    if normalized not in METHOD_ALIASES:
        raise ValueError(
            f"Unsupported method '{method_name}'. "
            f"Supported values: {', '.join(SUPPORTED_METHODS)}"
        )
    return METHOD_ALIASES[normalized]


def parse_methods(methods_value: str | Iterable[str]) -> List[str]:
    if isinstance(methods_value, str):
        items = [item.strip() for item in methods_value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in methods_value if str(item).strip()]

    methods = []
    for item in items:
        canonical = canonicalize_method_name(item)
        if canonical not in methods:
            methods.append(canonical)
    return methods


def diagnostic_alpha_grid(alpha: float) -> List[float]:
    candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    if 0.0 < float(alpha) < 1.0:
        candidates.append(float(alpha))
    return sorted({float(value) for value in candidates if 0.0 < float(value) < 1.0})


def save_placeholder_plot(path: str, title: str, message: str) -> None:
    plt.figure()
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def method_summary(
    p_values: List[float],
    n_attempted: int,
    alpha: float,
    metric_name: str,
    rate_denominator: str = "attempted",
) -> Dict:
    arr = np.asarray(p_values, dtype=float)
    n_pvalues = int(len(arr))
    n_attempted = int(n_attempted)
    n_missing = max(0, int(n_attempted) - n_pvalues)
    missing_rate = None if n_attempted == 0 else float(n_missing / n_attempted)
    n_rejections = int(np.sum(arr < alpha))
    conditional_rate = None if n_pvalues == 0 else float(n_rejections / n_pvalues)

    rate_denominator = str(rate_denominator).lower()
    if rate_denominator == "attempted":
        rate_value = None if n_attempted == 0 else float(n_rejections / n_attempted)
    elif rate_denominator == "pvalues":
        rate_value = conditional_rate
    else:
        raise ValueError("rate_denominator must be either 'attempted' or 'pvalues'.")

    def _binomial_se(p_hat, n):
        if p_hat is None or n is None or n <= 0:
            return None
        return float(np.sqrt(p_hat * (1.0 - p_hat) / n))

    rate_se = _binomial_se(rate_value, n_attempted if rate_denominator == "attempted" else n_pvalues)
    conditional_se = _binomial_se(conditional_rate, n_pvalues)

    return {
        "n_attempted": n_attempted,
        "n_pvalues": n_pvalues,
        "n_missing": n_missing,
        "missing_rate": missing_rate,
        "n_rejections": n_rejections,
        str(metric_name): rate_value,
        f"{metric_name}_se": rate_se,
        f"conditional_{metric_name}": conditional_rate,
        f"conditional_{metric_name}_se": conditional_se,
        "rate_denominator": rate_denominator,
        "uncertainty_source": "fixed_model_resampling_dependent",
    }


def save_method_outputs(
    results_dir: str,
    method: str,
    p_values: List[float],
    n_attempted: int,
    alpha: float,
    metric_name: str,
    rate_denominator: str = "attempted",
):
    method_dir = os.path.join(results_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    txt_path = os.path.join(method_dir, "p_values.txt")
    npy_path = os.path.join(method_dir, "p_values.npy")
    hist_path = os.path.join(method_dir, "histogram.png")
    ks_path = os.path.join(method_dir, "ks_test.txt")
    qq_path = os.path.join(method_dir, "qq_plot.png")
    calibration_plot_path = os.path.join(method_dir, "calibration_plot.png")
    calibration_json_path = os.path.join(method_dir, "calibration_curve.json")
    summary_path = os.path.join(method_dir, "summary.json")

    with open(txt_path, "w", encoding="utf-8") as handle:
        for p_value in p_values:
            handle.write(f"{p_value}\n")
    np.save(npy_path, np.array(p_values, dtype=float))

    arr = np.asarray(p_values, dtype=float)

    if len(arr) > 0:
        plt.figure()
        plt.hist(arr, bins=min(30, max(10, len(arr) // 10)))
        plt.title(f"Histogram of p-values ({method})")
        plt.xlabel("p-value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
    else:
        save_placeholder_plot(hist_path, f"Histogram of p-values ({method})", "No p-values")

    if len(arr) > 0:
        sorted_p = np.sort(arr)
        theoretical = (np.arange(1, len(sorted_p) + 1, dtype=float) - 0.5) / float(len(sorted_p))
        plt.figure()
        plt.plot(theoretical, sorted_p, marker="o", linestyle="", markersize=3)
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0)
        plt.title(f"Uniform QQ plot ({method})")
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Observed p-values")
        plt.tight_layout()
        plt.savefig(qq_path)
        plt.close()
    else:
        save_placeholder_plot(qq_path, f"Uniform QQ plot ({method})", "No p-values")

    alpha_grid = diagnostic_alpha_grid(alpha)
    calibration_curve = []
    for alpha_level in alpha_grid:
        n_rejections = int(np.sum(arr < alpha_level))
        calibration_curve.append(
            {
                "alpha": float(alpha_level),
                "empirical_rejection": None if int(n_attempted) == 0 else float(n_rejections / int(n_attempted)),
                "conditional_rejection": None if len(arr) == 0 else float(n_rejections / len(arr)),
            }
        )

    with open(calibration_json_path, "w", encoding="utf-8") as handle:
        json.dump(calibration_curve, handle, ensure_ascii=False, indent=2)

    if int(n_attempted) > 0:
        plt.figure()
        plt.plot(
            [item["alpha"] for item in calibration_curve],
            [item["empirical_rejection"] for item in calibration_curve],
            marker="o",
            label="headline",
        )
        if len(arr) > 0:
            plt.plot(
                [item["alpha"] for item in calibration_curve],
                [item["conditional_rejection"] for item in calibration_curve],
                marker="s",
                label="conditional",
            )
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0)
        plt.xlim(0.0, 0.55)
        plt.ylim(0.0, 0.55)
        plt.title(f"Calibration curve ({method})")
        plt.xlabel("Nominal alpha")
        plt.ylabel("Empirical rejection")
        plt.legend()
        plt.tight_layout()
        plt.savefig(calibration_plot_path)
        plt.close()
    else:
        save_placeholder_plot(calibration_plot_path, f"Calibration curve ({method})", "No p-values")

    if len(arr) > 0:
        ks_result = kstest(arr, "uniform")
        ks_stat = float(ks_result.statistic)
        ks_p = float(ks_result.pvalue)
        ks_text = f"KS test against uniform distribution: {ks_result}\n"
    else:
        ks_stat = None
        ks_p = None
        ks_text = "KS test against uniform distribution: skipped (no p-values)\n"

    with open(ks_path, "w", encoding="utf-8") as handle:
        handle.write(ks_text)

    summary = {
        "method": str(method),
        "metric_name": str(metric_name),
        **method_summary(
            p_values,
            n_attempted=n_attempted,
            alpha=alpha,
            metric_name=metric_name,
            rate_denominator=rate_denominator,
        ),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "histogram_path": hist_path,
        "qq_plot_path": qq_path,
        "calibration_plot_path": calibration_plot_path,
        "calibration_curve_path": calibration_json_path,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
