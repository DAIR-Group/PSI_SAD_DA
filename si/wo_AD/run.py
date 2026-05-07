import numpy as np
import torch

from ..detection import get_j_in_topk_intervals
from ..dnn.dnn import get_model_intervals as get_model_intervals_cpu
from ..dnn.util import parse_model
from ..util import (
    build_initial_sign_interval,
    build_test_reference_contrast_path,
    gen_data,
    load_models,
    resolve_source_target_test_sizes,
    truncated_cdf,
)


def apply_observed_linear_layer(intervals, params):
    W, bias = params
    new_intervals = []
    for left, right, a_curr, b_curr in intervals:
        a_new = a_curr @ W
        if bias is not None:
            a_new = a_new + bias
        b_new = b_curr @ W
        new_intervals.append((left, right, a_new, b_new))
    return new_intervals


def apply_observed_relu_layer(intervals, z_obs):
    new_intervals = []
    for left, right, a_curr, b_curr in intervals:
        pre_activation = a_curr + b_curr * float(z_obs)
        active_mask = (pre_activation > 0.0).astype(np.float64)
        new_intervals.append((left, right, a_curr * active_mask, b_curr * active_mask))
    return new_intervals


def apply_observed_leaky_relu_layer(intervals, params, z_obs):
    alpha = float(params)
    new_intervals = []
    for left, right, a_curr, b_curr in intervals:
        pre_activation = a_curr + b_curr * float(z_obs)
        slopes = np.where(pre_activation > 0.0, 1.0, alpha)
        new_intervals.append((left, right, a_curr * slopes, b_curr * slopes))
    return new_intervals


def apply_observed_batchnorm1d_layer(intervals, params):
    gamma, beta, running_mean, running_var, bn_eps = params
    new_intervals = []

    if running_var is None or running_mean is None:
        _, _, a0, _ = intervals[0]
        feature_dim = int(a0.shape[1])
        scale = np.ones(feature_dim, dtype=np.float64)
    else:
        scale = 1.0 / np.sqrt(running_var + bn_eps)

    if gamma is not None:
        scale = gamma * scale
    shift = np.zeros_like(scale) if running_mean is None else -running_mean * scale
    if beta is not None:
        shift = shift + beta

    for left, right, a_curr, b_curr in intervals:
        a_new = scale * a_curr + shift
        b_new = scale * b_curr
        new_intervals.append((left, right, a_new, b_new))
    return new_intervals


def get_observed_model_intervals(model, intervals, z_obs):
    layers = parse_model(model)

    for layer_type, params in layers:
        if layer_type == "Linear":
            intervals = apply_observed_linear_layer(intervals, params)
        elif layer_type == "ReLU":
            intervals = apply_observed_relu_layer(intervals, z_obs)
        elif layer_type == "LeakyReLU":
            intervals = apply_observed_leaky_relu_layer(intervals, params, z_obs)
        elif layer_type == "BatchNorm1d":
            intervals = apply_observed_batchnorm1d_layer(intervals, params)

    return sorted(intervals, key=lambda x: x[0])


def get_da_component(model):
    return getattr(model, "generator", None)


def get_ad_component(model):
    return getattr(model, "encoder", model)


def get_model_intervals_wo_ad_conditioning(model, intervals, z_obs):
    da_component = get_da_component(model)
    if da_component is not None:
        intervals = get_model_intervals_cpu(da_component, intervals)
    intervals = get_observed_model_intervals(get_ad_component(model), intervals, z_obs)
    return sorted(intervals, key=lambda x: x[0])


def run(
    seed: int,
    delta: float = 0,
    n: int = 150,
    target_mu: float = 0,
    source_mu: float = 2,
    d: int = 8,
    anomaly_rate: float = 0.0,
    top_k_percent: float = 0.05,
    top_k_normal_percent: float = 0.3,
    deepsad_encoder=None,
    deepsad_c=None,
    device: str = "auto",
    test_index_class: str = "normal",
    Sigma: np.ndarray = None,
    Sigma_source: np.ndarray | None = None,
    Sigma_target: np.ndarray | None = None,
    Sigma_ref: np.ndarray | None = None,
    X_source_obs: np.ndarray | None = None,
    target_rho: float = 0.0,
    source_rho: float = 0.0,
    reference_size: int = 100,
    source_test_size: int | None = None,
    target_test_size: int | None = None,
):
    _ = (
        top_k_normal_percent,
        X_source_obs,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)

    if test_index_class not in {"normal", "anomaly"}:
        raise ValueError("test_index_class must be either 'normal' or 'anomaly'.")
    _, n_source, n_target = resolve_source_target_test_sizes(
        n,
        source_test_size=source_test_size,
        target_test_size=target_test_size,
    )

    requested_device = "auto" if device is None else str(device).lower()
    if requested_device not in {"auto", "cpu", "cuda", "dnn_para"}:
        raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, dnn_para.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    if deepsad_encoder is None:
        load_device = None if requested_device == "auto" else requested_device
        deepsad_encoder, deepsad_c, _ = load_models(device=load_device)

    model_device = next(deepsad_encoder.parameters()).device
    target_model_device = requested_device
    if requested_device in {"auto", "dnn_para"}:
        target_model_device = model_device.type
    if target_model_device in {"cpu", "cuda"} and model_device.type != target_model_device:
        deepsad_encoder = deepsad_encoder.to(target_model_device)
        model_device = next(deepsad_encoder.parameters()).device

    def covariance_from_rho(rho_value: float) -> np.ndarray:
        if abs(float(rho_value)) < 1e-12:
            return np.eye(int(d), dtype=np.float64)
        idx = np.arange(int(d))
        return float(rho_value) ** np.abs(np.subtract.outer(idx, idx))

    if Sigma_target is None and Sigma is not None:
        Sigma_target = np.asarray(Sigma, dtype=np.float64)
    if Sigma_source is None:
        Sigma_source = covariance_from_rho(source_rho)
    if Sigma_target is None:
        Sigma_target = covariance_from_rho(target_rho)
    if Sigma_ref is None:
        Sigma_ref = Sigma_target

    if test_index_class == "normal":
        test_delta = 0.0
        test_anomaly_rate = 0.0
        candidate_label = 1
    else:
        test_delta = float(delta)
        test_anomaly_rate = float(anomaly_rate)
        candidate_label = -1

    X_source, y_source = gen_data(
        source_mu, test_delta, n_source, d, test_anomaly_rate, rho=source_rho, seed=seed
    )
    X_target, y_target = gen_data(
        target_mu, test_delta, n_target, d, test_anomaly_rate, rho=target_rho, seed=seed + 1
    )
    X_ref = gen_data(target_mu, 0.0, reference_size, d, 0.0, rho=target_rho, seed=seed + 2)[0]
    if not (np.all(np.isin(y_source, (-1, 1))) and np.all(np.isin(y_target, (-1, 1)))):
        raise ValueError("true_y contains invalid labels; expected values in {-1,1}.")

    X_detect = np.vstack([X_source, X_target])
    deepsad_c = np.asarray(deepsad_c, dtype=np.float64)
    with torch.no_grad():
        x_tensor = torch.tensor(X_detect, dtype=torch.float32, device=model_device)
        embeddings = deepsad_encoder(x_tensor).detach().cpu().numpy()
    scores = np.linalg.norm(embeddings - deepsad_c, axis=1)
    top_k = max(1, int(top_k_percent * len(scores)))
    top_k = min(top_k, len(scores))
    O = sorted(np.argpartition(scores, -top_k)[-top_k:])

    candidates = [
        int(i)
        for i in O
        if int(i) >= n_source and y_target[int(i) - n_source] == candidate_label
    ]
    if len(candidates) == 0:
        print(f"No '{test_index_class}' points for seed {seed}, skipping...")
        return []
    j_global = int(np.random.choice(candidates))
    j_target = j_global - n_source

    path = build_test_reference_contrast_path(
        X_test=X_target,
        X_ref=X_ref,
        j=j_target,
        X_source=X_source,
        Sigma_source=Sigma_source,
        Sigma_target=Sigma_target,
        Sigma_ref=Sigma_ref,
    )
    left, right = build_initial_sign_interval(
        path["a_test"],
        path["b_test"],
        j_target,
        path["positive_sign"],
        path["reference_a_mean"],
        path["reference_b_mean"],
        path["sigma"],
    )
    if left > right or not (left <= path["test_statistic"] <= right):
        return [0.0]

    a_detect = np.vstack([path["a_source"], path["a_test"]])
    b_detect = np.vstack([path["b_source"], path["b_test"]])


    intervals = get_model_intervals_wo_ad_conditioning(
        deepsad_encoder,
        [(left, right, a_detect, b_detect)],
        z_obs=path["test_statistic"],
    )
    intervals = get_j_in_topk_intervals(
        intervals, top_k_percent=top_k_percent, deepsad_c=deepsad_c, j=j_global
    )

    final_intervals = [(left_i, right_i, bool(Oz)) for left_i, right_i, Oz in intervals]
    cdf = truncated_cdf(0, path["sigma"], final_intervals, True, path["test_statistic"])
    if cdf is None:
        print(f"Warning: CDF computation failed for seed {seed}. Skipping this run.")
        return [None]
    p_value = 2 * min(cdf, 1 - cdf)
    print(f"p-value for seed {seed}: {p_value}")
    return [p_value]
