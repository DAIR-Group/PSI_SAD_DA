import argparse
import json
import os
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_sad import MLP_Autoencoder, AETrainer, DeepSADTrainer, MLP
from domain_adaptation import DADeepSADEncoder, WDGRL, WDGRLConfig


def resolve_rng(seed: int | None = None, rng=None):
    if rng is not None:
        return rng
    if seed is None:
        return np.random.RandomState()
    return np.random.RandomState(int(seed))


def make_synthetic_covariance(
    d: int,
    rho: float = 0.0,
    extra_noise_std: float = 0.0,
) -> np.ndarray:
    if abs(float(rho)) < 1e-12:
        cov = np.eye(int(d), dtype=float)
    else:
        idx = np.arange(int(d))
        cov = float(rho) ** np.abs(np.subtract.outer(idx, idx))

    if float(extra_noise_std) > 0.0:
        cov = cov + (float(extra_noise_std) ** 2) * np.eye(int(d), dtype=float)
    return np.asarray(cov, dtype=float)


def sample_synthetic_domain(
    mu: float,
    delta: float,
    n: int,
    d: int,
    anomaly_rate: float = 0.05,
    rho: float = 0.0,
    seed: Optional[int] = None,
    rng=None,
    covariance_noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = resolve_rng(seed=seed, rng=rng)
    cov = make_synthetic_covariance(
        d=d,
        rho=rho,
        extra_noise_std=covariance_noise_std,
    )

    X = rng.multivariate_normal(
        mean=np.full(d, mu, dtype=float),
        cov=cov,
        size=n,
    )
    true_labels = np.zeros(n, dtype=int)

    n_anomalies = int(n * anomaly_rate)
    if n_anomalies > 0:
        anomaly_idx = rng.choice(n, n_anomalies, replace=False)
        directions = rng.choice([-1, 1], size=(n_anomalies, d))
        X[anomaly_idx] += float(delta) * directions
        true_labels[anomaly_idx] = 1

    return X, true_labels


def sample_known_labels(
    true_labels: np.ndarray,
    known_label_rate: float,
    rng,
) -> np.ndarray:
    true_labels = np.asarray(true_labels, dtype=int)
    known_labels = np.zeros(len(true_labels), dtype=int)
    anomaly_idx = np.where(true_labels == 1)[0]
    normal_idx = np.where(true_labels == 0)[0]

    n_known_anom = int(len(anomaly_idx) * known_label_rate)
    if n_known_anom > 0:
        kn_anom_idx = rng.choice(anomaly_idx, n_known_anom, replace=False)
        known_labels[kn_anom_idx] = -1

    n_known_norm = int(len(normal_idx) * known_label_rate)
    if n_known_norm > 0:
        kn_norm_idx = rng.choice(normal_idx, n_known_norm, replace=False)
        known_labels[kn_norm_idx] = 1

    return known_labels


def gen_data(
    mu: float,
    delta: float,
    n: int,
    d: int,
    anomaly_rate: float = 0.05,
    known_label_rate: float = 0.2,
    rho: float = 0.0,
    seed: Optional[int] = None,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = resolve_rng(seed=seed, rng=rng)
    X, true_labels = sample_synthetic_domain(
        mu=mu,
        delta=delta,
        n=n,
        d=d,
        anomaly_rate=anomaly_rate,
        rho=rho,
        rng=rng,
    )
    known_labels = sample_known_labels(true_labels, known_label_rate, rng)

    return X, true_labels, known_labels


def build_synthetic_source_target_split(
    mu: float,
    delta: float,
    n_target_train: int,
    d: int,
    anomaly_rate: float,
    known_label_rate: float,
    rho: float,
    n_reference: int,
    seed: int | None,
    source_n: int | None = None,
    source_mean_shift: float = 0.5,
    source_noise_std: float = 0.0,
    source_delta: float | None = None,
    source_anomaly_rate: float | None = None,
    source_rho: float | None = None,
    target_mu: float | None = None,
    source_mu: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    dict,
]:
    n_total = int(n_target_train)
    if n_total <= 1:
        raise ValueError("--n must be greater than 1 for source-target synthetic training.")

    base_seed = 0 if seed is None else int(seed)
    target_rng = np.random.RandomState(base_seed)
    source_seed = base_seed + 10_000
    n_source = int(round(0.7 * n_total)) if source_n is None else int(source_n)
    n_source = min(max(1, n_source), n_total - 1)
    n_target = n_total - n_source
    if n_target <= 0:
        raise ValueError("--source-n must leave at least one target training point.")

    target_mu_value = float(mu if target_mu is None else target_mu)
    source_mu_value = float(target_mu_value + float(source_mean_shift) if source_mu is None else source_mu)
    source_rho_value = float(rho if source_rho is None else source_rho)
    source_delta_value = float(delta if source_delta is None else source_delta)
    source_anomaly_rate_value = float(anomaly_rate if source_anomaly_rate is None else source_anomaly_rate)

    X_target_train, y_target_train = sample_synthetic_domain(
        mu=target_mu_value,
        delta=delta,
        n=n_target,
        d=d,
        anomaly_rate=anomaly_rate,
        rho=rho,
        rng=target_rng,
    )
    known_y_target_train = sample_known_labels(
        y_target_train,
        known_label_rate=known_label_rate,
        rng=target_rng,
    )

    X_source, y_source = sample_synthetic_domain(
        mu=source_mu_value,
        delta=source_delta_value,
        n=n_source,
        d=d,
        anomaly_rate=source_anomaly_rate_value,
        rho=source_rho_value,
        seed=source_seed,
        covariance_noise_std=float(source_noise_std),
    )

    metadata = {
        "synthetic_data_layout": "source_target_train_only_v1",
        "train_has_reference_split": False,
        "source_target_ratio": {
            "source": float(n_source / n_total),
            "target": float(n_target / n_total),
        },
        "source_domain": {
            "n": int(len(X_source)),
            "mu": source_mu_value,
            "mean_shift_from_target": float(source_mu_value - target_mu_value),
            "delta": source_delta_value,
            "anomaly_rate": source_anomaly_rate_value,
            "rho": float(source_rho_value),
            "covariance_noise_std": float(source_noise_std),
            "label_policy": "source_labels_available_for_detector_filtering",
            "n_anomaly": int(np.sum(y_source == 1)),
            "n_normal": int(np.sum(y_source == 0)),
        },
        "target_domain": {
            "pool_n": int(n_target),
            "train_n": int(len(X_target_train)),
            "reference_n": 0,
            "mu": target_mu_value,
            "delta": float(delta),
            "anomaly_rate": float(anomaly_rate),
            "rho": float(rho),
            "known_label_rate": float(known_label_rate),
            "n_train_known_labeled": int(np.sum(known_y_target_train != 0)),
            "n_train_known_normal": int(np.sum(known_y_target_train == 1)),
            "n_train_known_anomaly": int(np.sum(known_y_target_train == -1)),
            "n_train_unlabeled": int(np.sum(known_y_target_train == 0)),
            "n_train_true_normal": int(np.sum(y_target_train == 0)),
            "n_train_true_anomaly": int(np.sum(y_target_train == 1)),
            "reference_policy": "not_used_in_train_generated_at_testing",
        },
        "split_seed": None if seed is None else int(seed),
        "source_seed": int(source_seed),
    }

    return (
        X_target_train,
        y_target_train,
        known_y_target_train,
        None,
        None,
        np.asarray(X_source, dtype=np.float64),
        np.asarray(y_source, dtype=int),
        metadata,
    )


def create_dataloader(
    X,
    true_labels,
    semi_labels,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    generator=None,
):
    semi_labels_processed = semi_labels.copy()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(true_labels, dtype=torch.long),
        torch.tensor(semi_labels_processed, dtype=torch.long),
        torch.arange(len(X)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )


def seed_everything(seed: int | None):
    if seed is None:
        return None

    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    return loader_generator


def build_training_metadata(
    input_dim: int,
    h_dims,
    rep_dim: int,
    da_enabled: bool,
    da_config: WDGRLConfig | None = None,
):
    metadata = {
        "model_type": "plain_deepsad",
        "da_enabled": False,
        "raw_input_dim": int(input_dim),
        "feature_dim_after_da": int(input_dim),
        "h_dims": [int(value) for value in h_dims],
        "rep_dim": int(rep_dim),
        "detector_training_input_mode": "target_raw_only",
    }
    if da_enabled and da_config is not None:
        metadata.update(
            {
                "model_type": "da_deepsad",
                "da_enabled": True,
                "feature_dim_after_da": int(da_config.generator_hidden_dims[-1]),
                "da": da_config.to_metadata(),
                "detector_training_input_mode": "source_plus_target_after_da",
            }
        )
    return metadata


def merge_da_detector_training_data(
    Z_source: np.ndarray,
    Z_target: np.ndarray,
    y_target: np.ndarray,
    known_y_target: np.ndarray,
    y_source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z_source = np.asarray(Z_source, dtype=np.float64)
    Z_target = np.asarray(Z_target, dtype=np.float64)
    y_target = np.asarray(y_target, dtype=int)
    known_y_target = np.asarray(known_y_target, dtype=int)

    if Z_source.ndim != 2 or Z_target.ndim != 2:
        raise ValueError("DA detector training merge expects 2D source and target feature arrays.")
    if Z_source.shape[1] != Z_target.shape[1]:
        raise ValueError(
            "DA detector training merge requires matching feature dimensions. "
            f"Got {Z_source.shape[1]} and {Z_target.shape[1]}."
        )

    n_source = int(len(Z_source))
    if n_source == 0:
        return Z_target, y_target, known_y_target

    if y_source is not None:
        y_source = np.asarray(y_source, dtype=int)
        if len(y_source) != n_source:
            raise ValueError(
                "Source labels must match source features. "
                f"Got {len(y_source)} labels for {n_source} source rows."
            )
        source_keep = y_source == 0
        Z_source = Z_source[source_keep]
        n_source = int(len(Z_source))
        if n_source == 0:
            return Z_target, y_target, known_y_target

    # Source contributes only normal unlabeled support to Deep SAD; source anomalies
    # are still available to WDGRL but are not pulled into the hypersphere center.
    y_source_detector = np.zeros(n_source, dtype=int)
    known_y_source = np.zeros(n_source, dtype=int)

    detector_input = np.concatenate([Z_source, Z_target], axis=0)
    detector_y = np.concatenate([y_source_detector, y_target], axis=0)
    detector_known_y = np.concatenate([known_y_source, known_y_target], axis=0)
    return detector_input, detector_y, detector_known_y


def train_detector_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    known_y_train: np.ndarray,
    d: int,
    h_dims,
    rep_dim: int,
    lr: float,
    ae_epochs: int,
    sad_epochs: int,
    batch_size: int,
    eta: float = 1.0,
    device: str | None = None,
    seed: int | None = None,
    X_source: np.ndarray | None = None,
    y_source: np.ndarray | None = None,
    enable_da: bool = False,
    da_generator_hidden_dims=None,
    da_critic_hidden_dims=None,
    da_gamma: float = 10.0,
    da_lr_generator: float | None = None,
    da_lr_critic: float | None = None,
    da_epochs: int = 50,
    da_critic_steps: int = 5,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.INFO)

    loader_generator = seed_everything(seed)
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    known_y_train = np.asarray(known_y_train, dtype=int)
    training_input = X_train
    detector_y = y_train
    detector_known_y = known_y_train
    training_input_dim = int(d)
    da_model = None
    da_config = None

    if enable_da:
        if X_source is None:
            raise ValueError("enable_da=True requires X_source for domain adaptation training.")

        X_source = np.asarray(X_source, dtype=np.float64)
        da_config = WDGRLConfig(
            generator_hidden_dims=tuple(
                int(value)
                for value in (da_generator_hidden_dims or [64, 32, rep_dim])
            ),
            critic_hidden_dims=tuple(
                int(value)
                for value in (da_critic_hidden_dims or [64, 32, 16])
            ),
            gamma=float(da_gamma),
            lr_generator=float(lr if da_lr_generator is None else da_lr_generator),
            lr_critic=float(lr if da_lr_critic is None else da_lr_critic),
            num_epochs=int(da_epochs),
            critic_steps=int(da_critic_steps),
            batch_size=int(batch_size),
        )
        da_model = WDGRL(
            input_dim=int(d),
            generator_hidden_dims=da_config.generator_hidden_dims,
            critic_hidden_dims=da_config.critic_hidden_dims,
            gamma=da_config.gamma,
            lr_generator=da_config.lr_generator,
            lr_critic=da_config.lr_critic,
            device=device,
        )

        source_loader = WDGRL.build_loader(
            X_source,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
        )
        target_loader = WDGRL.build_loader(
            X_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
        )
        da_history = da_model.fit(
            source_loader=source_loader,
            target_loader=target_loader,
            num_epochs=da_config.num_epochs,
            critic_steps=da_config.critic_steps,
        )
        transformed_source = np.asarray(da_model.transform_numpy(X_source), dtype=np.float64)
        transformed_target = np.asarray(da_model.transform_numpy(X_train), dtype=np.float64)
        training_input, detector_y, detector_known_y = merge_da_detector_training_data(
            transformed_source,
            transformed_target,
            y_train,
            known_y_train,
            y_source=y_source,
        )
        training_input_dim = int(training_input.shape[1])
    else:
        da_history = []

    train_batch_size = min(batch_size, len(training_input))
    if train_batch_size < 2:
        raise ValueError(
            "Training requires at least 2 samples to support BatchNorm. "
            f"Got {len(training_input)} samples."
        )

    train_loader = create_dataloader(
        training_input,
        detector_y,
        detector_known_y,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )

    ae_net = MLP_Autoencoder(
        x_dim=training_input_dim,
        h_dims=h_dims,
        rep_dim=rep_dim,
        bias=True,
    )
    ae_trainer = AETrainer(
        lr=lr,
        n_epochs=ae_epochs,
        batch_size=batch_size,
        device=device,
    )
    ae_net = ae_trainer.train(train_loader, ae_net)

    net = MLP(x_dim=training_input_dim, h_dims=h_dims, rep_dim=rep_dim, bias=True)
    net_dict = net.state_dict()
    ae_net_dict = ae_net.state_dict()
    ae_net_dict = {
        k.replace("encoder.", ""): v
        for k, v in ae_net_dict.items()
        if k.startswith("encoder.")
    }
    net_dict.update(ae_net_dict)
    net.load_state_dict(net_dict)

    sad_trainer = DeepSADTrainer(
        c=None,
        eta=eta,
        lr=lr,
        n_epochs=sad_epochs,
        batch_size=batch_size,
        device=device,
    )
    net = sad_trainer.train(train_loader, net)

    training_metadata = build_training_metadata(
        input_dim=int(d),
        h_dims=h_dims,
        rep_dim=rep_dim,
        da_enabled=bool(enable_da),
        da_config=da_config,
    )
    training_metadata["da_history"] = da_history
    training_metadata["n_target_train"] = int(len(X_train))
    training_metadata["n_detector_train_total"] = int(len(training_input))
    if enable_da and X_source is not None:
        training_metadata["n_source_train"] = int(len(X_source))
        training_metadata["source_training_label_policy"] = (
            "normal_source_points_unlabeled_support_source_anomalies_da_only"
            if y_source is not None
            else "all_source_points_treated_as_unlabeled"
        )
        if y_source is not None:
            y_source_arr = np.asarray(y_source, dtype=int)
            training_metadata["n_source_train_true_normal"] = int(np.sum(y_source_arr == 0))
            training_metadata["n_source_train_true_anomaly"] = int(np.sum(y_source_arr == 1))

    if enable_da and da_model is not None:
        encoder = DADeepSADEncoder(da_model.generator, net).to(device)
        encoder.training_metadata = dict(training_metadata)
        sad_trainer.training_metadata = dict(training_metadata)
        return encoder, sad_trainer, device

    net.training_metadata = dict(training_metadata)
    sad_trainer.training_metadata = dict(training_metadata)
    return net, sad_trainer, device


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)
    synthetic_split_metadata = None
    y_source = None

    base_seed = None if args.seed is None else int(args.seed)
    n_reference = args.n if args.n_reference is None else int(args.n_reference)
    (
        Xt,
        true_yt,
        known_yt,
        X_reference,
        reference_mean,
        X_source,
        y_source,
        synthetic_split_metadata,
    ) = build_synthetic_source_target_split(
        mu=args.mu,
        delta=args.delta,
        n_target_train=args.n,
        d=args.d,
        anomaly_rate=args.anomaly_rate,
        known_label_rate=args.known_label_rate,
        rho=args.rho if args.target_rho is None else args.target_rho,
        n_reference=n_reference,
        seed=base_seed,
        source_n=args.source_n,
        source_mean_shift=args.source_mean_shift,
        source_noise_std=args.source_noise_std,
        source_delta=args.source_delta,
        source_anomaly_rate=args.source_anomaly_rate,
        source_rho=args.source_rho,
        target_mu=args.target_mu,
        source_mu=args.source_mu,
    )

    h_dims = [int(x) for x in args.h_dims.split(",")]

    net, sad_trainer, device = train_detector_from_arrays(
        X_train=Xt,
        y_train=true_yt,
        known_y_train=known_yt,
        d=args.d,
        h_dims=h_dims,
        rep_dim=args.rep_dim,
        lr=args.lr,
        ae_epochs=args.ae_epochs,
        sad_epochs=args.sad_epochs,
        batch_size=args.batch_size,
        eta=args.eta,
        device=device,
        seed=args.seed,
        X_source=X_source,
        y_source=y_source,
        enable_da=args.enable_da,
        da_generator_hidden_dims=[int(x) for x in args.da_generator_hidden_dims.split(",") if x.strip()],
        da_critic_hidden_dims=[int(x) for x in args.da_critic_hidden_dims.split(",") if x.strip()],
        da_gamma=args.da_gamma,
        da_lr_generator=args.da_lr_generator,
        da_lr_critic=args.da_lr_critic,
        da_epochs=args.da_epochs,
        da_critic_steps=args.da_critic_steps,
    )

    if synthetic_split_metadata is not None:
        for holder in (net, sad_trainer):
            metadata = dict(getattr(holder, "training_metadata", {}))
            metadata["synthetic_split"] = synthetic_split_metadata
            metadata["n_target_reference"] = 0 if X_reference is None else int(len(X_reference))
            metadata["n_target_train_known_labeled"] = int(np.sum(known_yt != 0))
            metadata["n_target_train_unlabeled"] = int(np.sum(known_yt == 0))
            holder.training_metadata = metadata

    return (
        net,
        sad_trainer,
        device,
        Xt,
        true_yt,
        known_yt,
        X_reference,
        reference_mean,
        X_source,
    )


def evaluate(args, net, sad_trainer, device):
    Xt_test, yt_true_test, yt_known_test = gen_data(
        args.mu if args.target_mu is None else args.target_mu,
        args.delta,
        args.n_test,
        args.d,
        args.anomaly_rate,
        args.known_label_rate,
        rho=args.rho if args.target_rho is None else args.target_rho,
        seed=None if args.seed is None else int(args.seed) + 2,
    )
    test_loader = create_dataloader(
        Xt_test,
        yt_true_test,
        yt_known_test,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_labels, test_scores = sad_trainer.test(test_loader, net)

    auc = roc_auc_score(test_labels, test_scores)
    print(f"Test AUC: {auc:.4f}")
    return auc


def save_model(
    args,
    net,
    sad_trainer,
    Xt,
    known_yt=None,
    X_reference: np.ndarray | None = None,
    reference_mean: np.ndarray | None = None,
    X_source: np.ndarray | None = None,
):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.covariance_dir, exist_ok=True)

    covariance_source = "train.save_model"
    reference_mean_path = None
    source_data_path = None
    source_covariance_path = None

    model_path = os.path.join(args.model_dir, f"{args.name}_model.pth")
    center_path = os.path.join(args.model_dir, f"{args.name}_c.pth")
    covariance_path = os.path.join(args.covariance_dir, f"{args.name}_cov.npy")
    metadata_path = os.path.join(args.model_dir, f"{args.name}_metadata.json")
    if X_reference is not None and reference_mean is not None:
        reference_mean_path = os.path.join(args.model_dir, f"{args.name}_reference_mean.npy")
    if X_source is not None:
        source_data_path = os.path.join(args.model_dir, f"{args.name}_source.npy")
        source_covariance_path = os.path.join(args.covariance_dir, f"{args.name}_source_cov.npy")

    torch.save(net.state_dict(), model_path)
    torch.save(sad_trainer.c, center_path)

    target_rho = float(args.rho if args.target_rho is None else args.target_rho)
    covariance = make_synthetic_covariance(int(args.d), rho=target_rho)
    covariance_source = "synthetic_theoretical_target_covariance"
    np.save(covariance_path, covariance)
    if X_source is not None:
        np.save(source_data_path, np.asarray(X_source, dtype=np.float64))
        source_rho = float(args.rho if args.source_rho is None else args.source_rho)
        source_covariance = make_synthetic_covariance(
            int(args.d),
            rho=source_rho,
            extra_noise_std=float(args.source_noise_std),
        )
        np.save(source_covariance_path, source_covariance)
    metadata = dict(getattr(net, "training_metadata", getattr(sad_trainer, "training_metadata", {})))
    target_mu = float(args.mu if args.target_mu is None else args.target_mu)
    source_mu = (
        float(target_mu + float(args.source_mean_shift))
        if args.source_mu is None
        else float(args.source_mu)
    )
    metadata.update(
        {
            "model_name": str(args.name),
            "feature_dim": int(args.d),
            "mu": target_mu,
            "target_mu": target_mu,
            "source_mu": source_mu,
            "delta": float(args.delta),
            "n_train": int(args.n),
            "rho": float(args.rho if args.target_rho is None else args.target_rho),
            "target_rho": float(args.rho if args.target_rho is None else args.target_rho),
            "source_rho": float(args.rho if args.source_rho is None else args.source_rho),
            "n_reference": None if X_reference is None else int(len(X_reference)),
            "covariance_path": covariance_path,
            "covariance_source": covariance_source,
            "target_covariance_path": covariance_path,
            "target_covariance_source": covariance_source,
            "source_covariance_path": source_covariance_path,
            "source_covariance_source": None
            if source_covariance_path is None
            else "synthetic_theoretical_source_covariance",
            "source_data_path": source_data_path,
            "source_data_source": None if source_data_path is None else "source_training_split",
            "reference_mean_path": reference_mean_path,
            "reference_mean_source": None if reference_mean_path is None else covariance_source,
            "artifact_version": "2026-04-si-semi-ad-da-v2-exact-md",
        }
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Center saved to {center_path}")
    print(f"Covariance saved to {covariance_path}")
    if source_data_path is not None:
        print(f"Source data saved to {source_data_path}")
    if source_covariance_path is not None:
        print(f"Source covariance saved to {source_covariance_path}")
    if reference_mean_path is not None:
        print(f"Reference mean saved to {reference_mean_path}")
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Deep-SAD with optional WDGRL domain adaptation"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the saved model (e.g. 'baseline_mu0_d8')",
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--target-mu", type=float, default=None)
    parser.add_argument("--source-mu", type=float, default=None)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--anomaly-rate", type=float, default=0.05)
    parser.add_argument("--known-label-rate", type=float, default=0.2)
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Synthetic covariance setting: rho=0 gives independent data, rho>0 gives AR(1) correlated data.",
    )
    parser.add_argument("--target-rho", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--h-dims", type=str, default="128, 64, 32", help="Hidden dims, comma-separated"
    )
    parser.add_argument("--rep-dim", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--ae-epochs", type=int, default=5)
    parser.add_argument("--sad-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument(
        "--n-reference",
        type=int,
        default=None,
        help="Deprecated for synthetic training; target reference is generated only during SI/testing.",
    )
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument(
        "--covariance-dir",
        type=str,
        default="covariances",
        help="Directory to save covariance matrices.",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--no-save", action="store_true", help="Skip saving model")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for synthetic data generation.",
    )
    parser.add_argument("--enable-da", action="store_true", help="Enable WDGRL domain adaptation before AE + DeepSAD.")
    parser.add_argument(
        "--da-generator-hidden-dims",
        type=str,
        default="32,16,8,4,2",
        help="Generator hidden dims for WDGRL, comma-separated.",
    )
    parser.add_argument(
        "--da-critic-hidden-dims",
        type=str,
        default="32,16,8,4,2",
        help="Critic hidden dims for WDGRL, comma-separated.",
    )
    parser.add_argument("--da-gamma", type=float, default=10.0, help="Gradient-penalty weight for WDGRL.")
    parser.add_argument("--da-lr-generator", type=float, default=None, help="Learning rate for the DA generator.")
    parser.add_argument("--da-lr-critic", type=float, default=None, help="Learning rate for the DA critic.")
    parser.add_argument("--da-epochs", type=int, default=50, help="Number of WDGRL epochs.")
    parser.add_argument("--da-critic-steps", type=int, default=5, help="Number of critic updates per generator step.")
    parser.add_argument(
        "--source-n",
        type=int,
        default=None,
        help="Synthetic source-domain size inside total --n. Defaults to round(0.7 * --n).",
    )
    parser.add_argument(
        "--source-mean-shift",
        "--da-source-mean-shift",
        dest="source_mean_shift",
        type=float,
        default=0.5,
        help=(
            "Mean shift applied to the synthetic source domain relative to target. "
            "The legacy --da-source-mean-shift alias is kept for compatibility."
        ),
    )
    parser.add_argument(
        "--source-noise-std",
        "--da-source-noise-std",
        dest="source_noise_std",
        type=float,
        default=0.0,
        help=(
            "Extra isotropic covariance noise for the synthetic source domain. "
            "The legacy --da-source-noise-std alias is kept for compatibility."
        ),
    )
    parser.add_argument(
        "--source-delta",
        type=float,
        default=None,
        help="Synthetic source-domain anomaly shift. Defaults to --delta.",
    )
    parser.add_argument(
        "--source-anomaly-rate",
        type=float,
        default=None,
        help="Synthetic source-domain anomaly rate. Defaults to --anomaly-rate.",
    )
    parser.add_argument(
        "--source-rho",
        type=float,
        default=None,
        help="Synthetic source-domain AR(1) rho. Defaults to the target --rho.",
    )
    args = parser.parse_args()

    net, sad_trainer, device, Xt, true_yt, known_yt, X_reference, reference_mean, X_source = train(args)

    if not args.no_eval:
        evaluate(args, net, sad_trainer, device)

    if not args.no_save:
        save_model(
            args,
            net,
            sad_trainer,
            Xt,
            known_yt,
            X_reference=X_reference,
            reference_mean=reference_mean,
            X_source=X_source,
        )


if __name__ == "__main__":
    main()

