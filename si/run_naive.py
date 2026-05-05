import mpmath as mp
import numpy as np
import torch

from .util import (
    build_test_reference_contrast_path,
    gen_data,
    load_models,
    resolve_source_target_test_sizes,
)

mp.mp.dps = 500


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
        X_source_obs,
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

    if path["etajTsigmaetaj"][0][0] <= 0:
        return []
    z = mp.mpf(path["test_statistic"]) / mp.sqrt(mp.mpf(path["etajTsigmaetaj"][0][0]))
    p_value = mp.erfc(abs(z) / mp.sqrt(2))
    p_value = min(mp.mpf(1), max(p_value, mp.mpf("1e-50")))

    print(f"p-value for seed {seed}: {mp.nstr(p_value, 10)}")
    return float(p_value)


