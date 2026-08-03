import json
import os
from typing import Tuple

import mpmath as mp
import numpy as np
import torch
from deep_sad import MLP
from domain_adaptation import DADeepSADEncoder, Generator

mp.mp.dps = 500


def resolve_rng(seed: int | None = None, rng=None):
    if rng is not None:
        return rng
    if seed is None:
        return np.random.RandomState()
    return np.random.RandomState(int(seed))


def gen_data(
    mu: float,
    delta: float,
    n: int,
    d: int,
    anomaly_rate: float = 0.05,
    rho: float = 0.0,
    seed: int | None = None,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = resolve_rng(seed=seed, rng=rng)
    if abs(float(rho)) < 1e-12:
        cov = np.eye(d, dtype=float)
    else:
        idx = np.arange(d)
        cov = float(rho) ** np.abs(np.subtract.outer(idx, idx))

    X = rng.multivariate_normal(
        mean=np.full(d, mu, dtype=float),
        cov=cov,
        size=n,
    )
    true_labels = np.ones(n, dtype=int)

    n_anomalies = int(n * anomaly_rate)
    anomaly_idx = rng.choice(n, n_anomalies, replace=False)
    normal_idx = np.setdiff1d(np.arange(n), anomaly_idx)

    if n_anomalies > 0:
        directions = rng.choice([-1, 1], size=(n_anomalies, d))
        X[anomaly_idx] += float(delta) * directions
        true_labels[anomaly_idx] = -1

    true_labels[normal_idx] = 1

    return X, true_labels


def load_models(
    device: str = None,
    model_dir: str = "models",
    model_name: str = "deepsad",
    d: int = 8,
    h_dims: list = None,
    rep_dim: int = 1,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if h_dims is None:
        h_dims = [128, 64, 32]

    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "h_dims" in metadata:
            h_dims = [int(value) for value in metadata["h_dims"]]
        if "rep_dim" in metadata:
            rep_dim = int(metadata["rep_dim"])
        if "feature_dim" in metadata:
            d = int(metadata["feature_dim"])

    model_type = None if metadata is None else metadata.get("model_type")
    if model_type == "da_deepsad":
        da_payload = dict(metadata.get("da", {}))
        generator_hidden_dims = [int(value) for value in da_payload.get("generator_hidden_dims", [32, 16, 8, 4, 2])]
        generator = Generator(input_dim=int(d), hidden_dims=generator_hidden_dims).to(device)
        inner_input_dim = int(generator_hidden_dims[-1])
        deepsad_backbone = MLP(
            x_dim=inner_input_dim,
            h_dims=h_dims,
            rep_dim=rep_dim,
            bias=True,
        ).to(device)
        deepsad_model = DADeepSADEncoder(generator, deepsad_backbone).to(device)
    else:
        deepsad_model = MLP(x_dim=d, h_dims=h_dims, rep_dim=rep_dim, bias=True).to(device)

    deepsad_state = torch.load(
        os.path.join(model_dir, f"{model_name}_model.pth"), map_location=device
    )
    deepsad_c = torch.load(
        os.path.join(model_dir, f"{model_name}_c.pth"), map_location=device
    )

    deepsad_model.load_state_dict(deepsad_state)
    deepsad_model.eval()

    deepsad_c = deepsad_c.detach().cpu().numpy()

    return deepsad_model, deepsad_c, device


def load_model_metadata(
    model_dir: str = "models",
    model_name: str = "deepsad",
) -> dict | None:
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_working_model_artifacts(
    model_dir: str = "models",
    model_name: str = "deepsad",
    allow_missing_source: bool = False,
) -> dict:
    metadata = load_model_metadata(model_dir=model_dir, model_name=model_name) or {}
    base_dir = os.path.dirname(os.path.abspath(model_dir))
    feature_dim = None if "feature_dim" not in metadata else int(metadata["feature_dim"])

    def resolve_artifact_path(path: str | None) -> str | None:
        if path is None:
            return None
        if os.path.exists(path):
            return path
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            local = os.path.join(base_dir, *parts[-2:])
            if os.path.exists(local):
                return local
        return path

    target_covariance_path = resolve_artifact_path(
        metadata.get("target_covariance_path") or metadata.get("covariance_path")
    )
    source_covariance_path = resolve_artifact_path(metadata.get("source_covariance_path"))
    source_data_path = resolve_artifact_path(metadata.get("source_data_path"))

    Sigma_target = None
    if target_covariance_path is not None and os.path.exists(target_covariance_path):
        Sigma_target = np.asarray(np.load(target_covariance_path), dtype=np.float64)

    Sigma_source = None
    if source_covariance_path is not None and os.path.exists(source_covariance_path):
        Sigma_source = np.asarray(np.load(source_covariance_path), dtype=np.float64)

    X_source = None
    if source_data_path is not None and os.path.exists(source_data_path):
        X_source = np.asarray(np.load(source_data_path), dtype=np.float64)
    elif feature_dim is not None:
        X_source = np.empty((0, feature_dim), dtype=np.float64)

    model_type = str(metadata.get("model_type", "plain_deepsad"))
    if model_type == "da_deepsad" and not allow_missing_source:
        if X_source is None:
            raise FileNotFoundError(
                f"Missing DA source artifact for model '{model_name}'. "
                "Retrain the model to produce source_data_path/source_covariance_path metadata."
            )
        if Sigma_source is None:
            raise FileNotFoundError(
                f"Missing DA source covariance artifact for model '{model_name}'. "
                "Retrain the model to produce source_data_path/source_covariance_path metadata."
            )

    if X_source is None:
        if feature_dim is None:
            raise ValueError(
                f"Could not infer feature_dim for model '{model_name}'. "
                "Expected metadata with feature_dim or source_data_path."
            )
        X_source = np.empty((0, feature_dim), dtype=np.float64)

    return {
        "metadata": metadata,
        "X_source": X_source,
        "Sigma_source": Sigma_source,
        "Sigma_target": Sigma_target, 
        "Sigma_ref": Sigma_target,
    }

def normal_interval_prob(left, right, mu, sigma):
    z_left = (left - mu) / sigma
    z_right = (right - mu) / sigma
    
    tail_left = 0.5 * mp.erfc(z_left / mp.sqrt(2))
    tail_right = 0.5 * mp.erfc(z_right / mp.sqrt(2))
    
    return tail_left - tail_right

def truncated_cdf(mu, sigma, intervals, O_or_etajTX, etajTX=None, obs_tol: float = 1e-10):
    if etajTX is None:
        observed_selection = None
        etajTX = O_or_etajTX
    else:
        observed_selection = O_or_etajTX

    numerator = 0
    denominator = 0
    for left, right, Oz in intervals:
        if observed_selection is not None and observed_selection != Oz:
            if (etajTX >= left - obs_tol) and (etajTX <= right + obs_tol):
                return None
            continue

        denominator = (
            denominator + normal_interval_prob(left, right, mu, sigma)
        )
        if etajTX >= right - obs_tol:
            numerator = (
                numerator + normal_interval_prob(left, right, mu, sigma)
            )
        if (etajTX >= left - obs_tol) and (etajTX <= right + obs_tol):
            clipped_obs = min(max(float(etajTX), float(left)), float(right))
            numerator = (
                numerator
                + normal_interval_prob(left, clipped_obs, mu, sigma)
            )
    if denominator != 0:
        return float(numerator / denominator)
    return None


def resolve_block_covariance(Sigma: np.ndarray | None, d: int) -> np.ndarray:
    if Sigma is None:
        return np.eye(int(d), dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    expected_shape = (int(d), int(d))
    if Sigma.shape != expected_shape:
        raise ValueError(f"Expected covariance shape {expected_shape}, got {Sigma.shape}.")
    return Sigma


def resolve_source_target_test_sizes(
    n: int,
    source_test_size: int | None = None,
    target_test_size: int | None = None,
) -> tuple[int, int, int]:
    n_total = int(n)
    source_given = source_test_size is not None
    target_given = target_test_size is not None

    if source_given:
        source_test_size = int(source_test_size)
        if source_test_size <= 0:
            raise ValueError("source_test_size must be positive when provided.")
    if target_given:
        target_test_size = int(target_test_size)
        if target_test_size <= 0:
            raise ValueError("target_test_size must be positive when provided.")

    if source_given and target_given:
        n_source = int(source_test_size)
        n_target = int(target_test_size)
        return n_source + n_target, n_source, n_target

    if n_total < 2:
        raise ValueError("n must be at least 2 so both source and target contain at least one point.")

    if source_given:
        n_source = int(source_test_size)
        if n_source >= n_total:
            raise ValueError("source_test_size must be smaller than n when target_test_size is not provided.")
        return n_total, n_source, n_total - n_source

    if target_given:
        n_target = int(target_test_size)
        if n_target >= n_total:
            raise ValueError("target_test_size must be smaller than n when source_test_size is not provided.")
        return n_total, n_total - n_target, n_target

    n_source = int(round(0.7 * n_total))
    n_source = min(max(1, n_source), n_total - 1)
    n_target = n_total - n_source
    return n_total, n_source, n_target


def prepare_si_model(deepsad_encoder, deepsad_c, device: str):
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

    return requested_device, model_device, deepsad_encoder, np.asarray(deepsad_c, dtype=np.float64)


def build_test_reference_contrast_path(
    X_test: np.ndarray,
    X_ref: np.ndarray,
    j: int,
    X_source: np.ndarray | None = None,
    Sigma_source: np.ndarray | None = None,
    Sigma_target: np.ndarray | None = None,
    Sigma_ref: np.ndarray | None = None,
) -> dict:
    X_test = np.asarray(X_test, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)
    if X_test.ndim != 2 or X_ref.ndim != 2:
        raise ValueError("X_test and X_ref must both be 2D arrays.")
    if len(X_ref) == 0:
        raise ValueError("X_ref must contain at least one reference sample.")

    d = int(X_test.shape[1])
    if X_ref.shape[1] != d:
        raise ValueError(
            f"Feature mismatch between X_test and X_ref: {X_test.shape[1]} vs {X_ref.shape[1]}."
        )

    if X_source is None:
        X_source = np.empty((0, d), dtype=np.float64)
    X_source = np.asarray(X_source, dtype=np.float64)
    if X_source.ndim != 2 or X_source.shape[1] != d:
        raise ValueError(
            f"X_source must be a 2D array with feature dimension {d}. Got shape {X_source.shape}."
        )

    Sigma_target = resolve_block_covariance(Sigma_target, d)
    Sigma_ref = resolve_block_covariance(Sigma_ref, d) if Sigma_ref is not None else Sigma_target
    Sigma_source = resolve_block_covariance(Sigma_source, d) if len(X_source) > 0 else np.eye(d, dtype=np.float64)

    n_source = int(len(X_source))
    n_test = int(len(X_test))
    n_ref = int(len(X_ref))

    X_ref_mean = np.mean(X_ref, axis=0)
    positive_sign = np.where(X_test[int(j)] - X_ref_mean >= 0.0, 1.0, -1.0).astype(np.float64)

    eta_source = np.zeros((n_source, d), dtype=np.float64)
    eta_test = np.zeros((n_test, d), dtype=np.float64)
    eta_test[int(j), :] = positive_sign
    eta_ref = np.tile((-positive_sign / float(n_ref)), (n_ref, 1))

    def apply_cov(eta_block: np.ndarray, Sigma_block: np.ndarray) -> np.ndarray:
        if eta_block.size == 0:
            return np.empty_like(eta_block)
        return eta_block @ Sigma_block.T

    Sigma_eta_source = apply_cov(eta_source, Sigma_source)
    Sigma_eta_test = apply_cov(eta_test, Sigma_target)
    Sigma_eta_ref = apply_cov(eta_ref, Sigma_ref)

    etajTsigmaetaj = (
        float(np.einsum("ij,ij->", eta_source, Sigma_eta_source))
        + float(np.einsum("ij,ij->", eta_test, Sigma_eta_test))
        + float(np.einsum("ij,ij->", eta_ref, Sigma_eta_ref))
    )
    if etajTsigmaetaj <= 0.0:
        raise ValueError("The test/reference contrast variance must be positive.")

    test_statistic = float(positive_sign @ (X_test[int(j)] - X_ref_mean))

    b_source = np.zeros_like(X_source, dtype=np.float64)
    if Sigma_eta_source.size > 0:
        b_source = Sigma_eta_source / etajTsigmaetaj
    b_test = Sigma_eta_test / etajTsigmaetaj
    b_ref = Sigma_eta_ref / etajTsigmaetaj

    a_source = np.asarray(X_source, dtype=np.float64) - b_source * test_statistic
    a_test = np.asarray(X_test, dtype=np.float64) - b_test * test_statistic
    a_ref = np.asarray(X_ref, dtype=np.float64) - b_ref * test_statistic

    eta_vector = np.concatenate([eta_source, eta_test, eta_ref], axis=0).reshape(-1)
    Y_vector = np.concatenate([X_source, X_test, X_ref], axis=0).reshape(-1)

    return {
        "n_source": n_source,
        "n_test": n_test,
        "n_ref": n_ref,
        "d": d,
        "X_ref_mean": X_ref_mean,
        "positive_sign": positive_sign,
        "test_statistic": test_statistic,
        "etajTx_raw": test_statistic,
        "etajTsigmaetaj": np.array([[etajTsigmaetaj]], dtype=np.float64),
        "sigma": float(np.sqrt(etajTsigmaetaj)),
        "eta_vector": eta_vector,
        "Y_vector": Y_vector,
        "a_source": a_source,
        "a_test": a_test,
        "a_ref": a_ref,
        "b_source": b_source,
        "b_test": b_test,
        "b_ref": b_ref,
        "reference_a_mean": np.mean(a_ref, axis=0),
        "reference_b_mean": np.mean(b_ref, axis=0),
    }


def build_initial_sign_interval(
    a_test: np.ndarray,
    b_test: np.ndarray,
    j: int,
    positive_sign: np.ndarray,
    reference_a_mean: np.ndarray,
    reference_b_mean: np.ndarray,
    sigma: float,
) -> tuple[float, float]:
    itv = [-20.0 * float(sigma), 20.0 * float(sigma)]
    for i in range(a_test.shape[1]):
        new_a = (a_test[int(j), i] - reference_a_mean[i]) * positive_sign[i]
        new_b = (b_test[int(j), i] - reference_b_mean[i]) * positive_sign[i]

        if abs(new_b) < 1e-16:
            if new_a < 0.0:
                return 1.0, 0.0
            continue

        z = -new_a / new_b
        if new_b > 0.0:
            itv = [max(itv[0], z), itv[1]]
        else:
            itv = [itv[0], min(itv[1], z)]

    left = float(itv[0].item() if isinstance(itv[0], np.ndarray) else itv[0])
    right = float(itv[1].item() if isinstance(itv[1], np.ndarray) else itv[1])
    return left, right

def compute_etajTsigmaetaj_a_b(etaj, etajTx, X, n, d, S=None):
    if S is None:
        S = np.eye(d)                          # special case: sigma = I_{n*d}
    etaj_blocks = etaj.reshape(n, d)           # (n, d)
    S_etaj = etaj_blocks @ S.T                 # (n, d)
    etajTsigmaetaj = np.einsum('ij,ij->', etaj_blocks, S_etaj).reshape(1, 1)
    
    
    S_etaj_flat = S_etaj.reshape(-1, 1)                                     # (n*d, 1)
    b = S_etaj_flat / etajTsigmaetaj                                        # (n*d, 1)

    X_flat = X.reshape(-1, 1)
    a = X_flat - b * etajTx    
    return etajTsigmaetaj, a, b

