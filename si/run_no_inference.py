import numpy as np
import torch

from .util import (
    gen_data,
    load_models,
    resolve_source_target_test_sizes,
)


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
    reference_size: int = 1000,
    source_test_size: int | None = None,
    target_test_size: int | None = None,
):
    _ = (
        top_k_normal_percent,
        Sigma,
        Sigma_source,
        Sigma_target,
        Sigma_ref,
        X_source_obs,
        reference_size,
    )
    np.random.seed(seed)
    torch.manual_seed(seed)

    if test_index_class not in {"normal", "anomaly"}:
        raise ValueError("test_index_class must be either 'normal' or 'anomaly'.")
    n_total, n_source, n_target = resolve_source_target_test_sizes(
        n,
        source_test_size=source_test_size,
        target_test_size=target_test_size,
    )

    requested_device = "auto" if device is None else str(device).lower()
    if requested_device not in {"auto", "cpu", "cuda", "dnn_para"}:
        raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, dnn_para.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if requested_device == "dnn_para":
        requested_device = "auto"

    if deepsad_encoder is None:
        load_device = None if requested_device == "auto" else requested_device
        deepsad_encoder, deepsad_c, _ = load_models(device=load_device)

    model_device = next(deepsad_encoder.parameters()).device
    target_model_device = model_device.type if requested_device == "auto" else requested_device
    if target_model_device in {"cpu", "cuda"} and model_device.type != target_model_device:
        deepsad_encoder = deepsad_encoder.to(target_model_device)
        model_device = next(deepsad_encoder.parameters()).device

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
    if not (np.all(np.isin(y_source, (-1, 1))) and np.all(np.isin(y_target, (-1, 1)))):
        raise ValueError("true_y contains invalid labels; expected values in {-1,1}.")

    X_detect = np.vstack([X_source, X_target])
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

    # Per spec, no-inference rejects whenever a target candidate is selected.
    return [0.0]
