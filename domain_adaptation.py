from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from itertools import cycle
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_hidden_dims(hidden_dims: Sequence[int] | None, default: Sequence[int]) -> list[int]:
    if hidden_dims is None:
        hidden_dims = default
    parsed = [int(value) for value in hidden_dims]
    if len(parsed) == 0:
        raise ValueError("hidden_dims must contain at least one value.")
    return parsed


def extract_tensor(batch) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
        prev_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def ensure_info_logging() -> logging.Logger:
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        if handler.level > logging.INFO:
            handler.setLevel(logging.INFO)
    return logger


@dataclass(frozen=True)
class WDGRLConfig:
    generator_hidden_dims: tuple[int, ...] = (32, 16, 8, 4, 2)
    critic_hidden_dims: tuple[int, ...] = (32, 16, 8, 4, 2)
    gamma: float = 10.0
    lr_generator: float = 1e-3
    lr_critic: float = 1e-3
    num_epochs: int = 50
    critic_steps: int = 5
    batch_size: int = 64

    def to_metadata(self) -> dict:
        return asdict(self)


class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        hidden_dims = parse_hidden_dims(hidden_dims, default=(32, 16, 8, 4, 2))
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.output_dim = int(hidden_dims[-1])
        self.net = build_mlp(self.input_dim, hidden_dims[:-1], output_dim=self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(int(x.size(0)), -1))


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        hidden_dims = parse_hidden_dims(hidden_dims, default=(32, 16, 8, 4, 2))
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.net = build_mlp(self.input_dim, hidden_dims, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(int(x.size(0)), -1))


class DADeepSADEncoder(nn.Module):
    def __init__(self, generator: Generator, encoder: nn.Module):
        super().__init__()
        self.generator = generator
        self.encoder = encoder
        self.rep_dim = getattr(encoder, "rep_dim", None)
        self.input_dim = getattr(generator, "input_dim", None)
        self.da_output_dim = getattr(generator, "output_dim", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.generator(x)
        return self.encoder(features)


class WDGRL:
    def __init__(
        self,
        input_dim: int,
        generator_hidden_dims: Sequence[int] | None = None,
        critic_hidden_dims: Sequence[int] | None = None,
        gamma: float = 10.0,
        lr_generator: float = 1e-3,
        lr_critic: float = 1e-3,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(input_dim, generator_hidden_dims or (32, 16, 8, 4, 2)).to(self.device)
        self.critic = Critic(self.generator.output_dim, critic_hidden_dims or (32, 16, 8, 4, 2)).to(self.device)
        self.gamma = float(gamma)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=float(lr_generator))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(lr_critic))

    def compute_gradient_penalty(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        if source_features.shape != target_features.shape:
            raise ValueError(
                "Gradient penalty requires source and target features with the same shape. "
                f"Got {tuple(source_features.shape)} and {tuple(target_features.shape)}."
            )

        alpha = torch.rand(source_features.size(0), 1, device=self.device, dtype=source_features.dtype)
        interpolates = alpha * source_features + (1.0 - alpha) * target_features
        interpolates.requires_grad_(True)

        critic_scores = self.critic(interpolates)
        gradients = torch.autograd.grad(
            outputs=critic_scores,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_scores),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1.0) ** 2).mean()

    def fit(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        num_epochs: int = 50,
        critic_steps: int = 5,
    ) -> list[dict]:
        logger = ensure_info_logging()
        self.generator.train()
        self.critic.train()
        history: list[dict] = []

        if len(source_loader) == 0 or len(target_loader) == 0:
            raise ValueError("WDGRL requires non-empty source and target loaders.")

        logger.info("Starting domain adaptation training...")
        start_time = time.time()
        num_batches = max(len(source_loader), len(target_loader))

        for epoch in range(int(num_epochs)):
            source_iter = cycle(source_loader)
            target_iter = cycle(target_loader)
            epoch_wasserstein = 0.0
            epoch_gp = 0.0
            epoch_generator_loss = 0.0

            for _ in range(num_batches):
                source_batch = extract_tensor(next(source_iter)).to(self.device)
                target_batch = extract_tensor(next(target_iter)).to(self.device)

                batch_size = min(int(source_batch.size(0)), int(target_batch.size(0)))
                if batch_size < 2:
                    continue

                source_batch = source_batch[:batch_size]
                target_batch = target_batch[:batch_size]

                for _ in range(int(critic_steps)):
                    self.critic_optimizer.zero_grad()
                    with torch.no_grad():
                        source_features = self.generator(source_batch)
                        target_features = self.generator(target_batch)

                    critic_source = self.critic(source_features)
                    critic_target = self.critic(target_features)
                    wasserstein_distance = critic_source.mean() - critic_target.mean()
                    gradient_penalty = self.compute_gradient_penalty(source_features, target_features)
                    critic_loss = -wasserstein_distance + self.gamma * gradient_penalty
                    critic_loss.backward()
                    self.critic_optimizer.step()

                self.generator_optimizer.zero_grad()
                source_features = self.generator(source_batch)
                target_features = self.generator(target_batch)
                generator_loss = self.critic(source_features).mean() - self.critic(target_features).mean()
                generator_loss.backward()
                self.generator_optimizer.step()

                epoch_wasserstein += float(generator_loss.detach().cpu().item())
                epoch_gp += float(gradient_penalty.detach().cpu().item())
                epoch_generator_loss += float(generator_loss.detach().cpu().item())

            history.append(
                {
                    "epoch": int(epoch + 1),
                    "wasserstein_distance": float(epoch_wasserstein / num_batches),
                    "gradient_penalty": float(epoch_gp / num_batches),
                    "generator_loss": float(epoch_generator_loss / num_batches),
                }
            )

            if (epoch + 1) % 20 == 0 or (epoch + 1) == int(num_epochs):
                logger.info(
                    "DA Epoch %03d/%03d | Wasserstein: %.6f | GP: %.6f | Generator Loss: %.6f",
                    epoch + 1,
                    int(num_epochs),
                    float(epoch_wasserstein / num_batches),
                    float(epoch_gp / num_batches),
                    float(epoch_generator_loss / num_batches),
                )

        self.generator.eval()
        self.critic.eval()
        train_time = time.time() - start_time
        logger.info("Domain adaptation finished in %.2fs", train_time)
        return history

    @torch.no_grad()
    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        return self.generator(x.to(self.device))

    @torch.no_grad()
    def transform_numpy(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D array for DA transform, got shape {x.shape}.")

        outputs = []
        self.generator.eval()
        for start in range(0, len(x), int(batch_size)):
            batch = torch.as_tensor(x[start : start + int(batch_size)], dtype=torch.float32, device=self.device)
            outputs.append(self.generator(batch).detach().cpu().numpy())
        if len(outputs) == 0:
            return np.empty((0, self.generator.output_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    @staticmethod
    def build_loader(
        X: np.ndarray,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
        return DataLoader(
            dataset,
            batch_size=min(int(batch_size), len(dataset)),
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
            generator=generator,
        )

