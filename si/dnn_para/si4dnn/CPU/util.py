import numpy as np
import torch
from torch import nn


def parse_torch_model(model: nn.Module):
    """
    Flatten model into a list of (layer_type, tensor_or_None)
    e.g. [("Weight", tensor), ("Bias", tensor), ("ReLU", None), ...]

    Args:
        model: PyTorch nn.Module
        to_numpy: If True, convert tensors to numpy arrays
    """
    results = []
    
    # print(model.modules())

    for module in model.modules():
        # print(f"Processing module: {module.__class__.__name__}")
        if isinstance(module, nn.Sequential) or module is model:
            continue  # skip containers

        # Special handling for LeakyReLU to extract negative_slope
        if isinstance(module, nn.LeakyReLU):
            alpha = module.negative_slope
            results.append(("LeakyReLU", alpha))
            continue

        # Special handling for BatchNorm1d (must come before generic weight/bias check)
        if isinstance(module, nn.BatchNorm1d):
            gamma = (
                module.weight.detach().cpu().numpy()
                if module.affine and module.weight is not None
                else None
            )
            beta = (
                module.bias.detach().cpu().numpy()
                if module.affine and module.bias is not None
                else None
            )
            running_mean = (
                module.running_mean.detach().cpu().numpy()
                if module.track_running_stats and module.running_mean is not None
                else None
            )
            running_var = (
                module.running_var.detach().cpu().numpy()
                if module.track_running_stats and module.running_var is not None
                else None
            )
            eps = module.eps
            results.append(
                ("BatchNorm1d", (gamma, beta, running_mean, running_var, eps))
            )
            continue

        # Handle parameterized modules (Linear, Conv, etc.)
        if hasattr(module, "weight") or hasattr(module, "bias"):
            w = (
                module.weight.detach()
                if hasattr(module, "weight") and module.weight is not None
                else None
            )
            b = (
                module.bias.detach()
                if hasattr(module, "bias") and module.bias is not None
                else None
            )

            w = w.cpu().numpy().T if w is not None else None
            b = b.cpu().numpy() if b is not None else None

            results.append((module.__class__.__name__, (w, b)))
            continue

        # Handle activation functions / parameterless modules
        if not any(p.requires_grad for p in module.parameters()):
            results.append((module.__class__.__name__, None))

    return results


def is_torch_model(model):
    return isinstance(model, nn.Module)


def parse_model(model):
    if is_torch_model(model):
        return parse_torch_model(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
