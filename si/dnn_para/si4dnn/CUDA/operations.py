import torch


def Linear(a, b, params):
    stacked = torch.stack([a, b], dim=0)  # shape: (2, ...)
    result = torch.matmul(stacked, params[0])  # shape: (2, ...)
    result[0] += params[1]
    return result[0], result[1]


def relu_elementwise(a, b, z):
    X = a + b * z
    neg_mask = X <= 0
    b_nz = torch.abs(b) > 1e-12
    a_out = torch.where(neg_mask, torch.tensor(0.0, device=a.device), a)
    b_out = torch.where(neg_mask, torch.tensor(0.0, device=b.device), b)
    threshold = torch.where(b_nz, -a / b, torch.tensor(float("inf"), device=a.device))
    where_min = (neg_mask & (b > 0)) | (~neg_mask & (b < 0))
    where_max = (neg_mask & (b < 0)) | (~neg_mask & (b > 0))
    return a_out, b_out, threshold, where_min, where_max


def ReLU(a, b, z, itv):
    a_out, b_out, threshold, where_min, where_max = relu_elementwise(a, b, z)
    min_val_array = torch.where(
        where_min, threshold, torch.tensor(float("inf"), device=threshold.device)
    )
    min_val = torch.min(min_val_array)
    max_val_array = torch.where(
        where_max, threshold, torch.tensor(float("-inf"), device=threshold.device)
    )
    max_val = torch.max(max_val_array)
    new_upper = torch.minimum(itv[1], min_val)
    new_lower = torch.maximum(itv[0], max_val)
    itv_out = torch.stack([new_lower, new_upper])
    itv_out = torch.where(
        new_lower <= new_upper, itv_out, torch.full_like(itv_out, float("nan"))
    )
    return a_out, b_out, itv_out


def leaky_relu_elementwise(a, b, z, alpha=0.01):
    """
    Helper function for LeakyReLU element-wise computation
    """
    X = a + b * z
    neg_mask = X <= 0
    b_nz = torch.abs(b) > 1e-12

    # Apply alpha scaling to negative region
    a_out = torch.where(neg_mask, alpha * a, a)
    b_out = torch.where(neg_mask, alpha * b, b)

    # Threshold computation for interval arithmetic
    threshold = torch.where(b_nz, -a / b, torch.tensor(float("inf"), device=a.device))

    # Determine where to update min/max bounds (same as ReLU)
    where_min = (neg_mask & (b > 0)) | (~neg_mask & (b < 0))
    where_max = (neg_mask & (b < 0)) | (~neg_mask & (b > 0))

    return a_out, b_out, threshold, where_min, where_max


def LeakyReLU(a, b, z, itv, alpha=0.01):
    """
    LeakyReLU activation: f(x) = max(alpha*x, x)

    Args:
        a: constant term (GPU tensor)
        b: coefficient term (GPU tensor)
        z: symbolic variable value (GPU tensor)
        itv: current interval bounds [lower, upper] (GPU tensor)
        alpha: negative slope (default 0.01)
    """
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError(f"LeakyReLU alpha must be non-negative, got {alpha}")

    a_out, b_out, threshold, where_min, where_max = leaky_relu_elementwise(
        a, b, z, alpha
    )

    # Update interval bounds
    min_val_array = torch.where(
        where_min, threshold, torch.tensor(float("inf"), device=threshold.device)
    )
    min_val = torch.min(min_val_array)
    max_val_array = torch.where(
        where_max, threshold, torch.tensor(float("-inf"), device=threshold.device)
    )
    max_val = torch.max(max_val_array)

    new_upper = torch.minimum(itv[1], min_val)
    new_lower = torch.maximum(itv[0], max_val)

    itv_out = torch.stack([new_lower, new_upper])
    itv_out = torch.where(
        new_lower <= new_upper, itv_out, torch.full_like(itv_out, float("nan"))
    )

    return a_out, b_out, itv_out


def BatchNorm1d(a, b, params):
    """
    BatchNorm1d: y = (x - mean) / sqrt(var + eps) * gamma + beta

    For symbolic form x = a + b * z:
        y = (a - mean) / std * gamma + beta + (b / std * gamma) * z

    Where std = sqrt(var + eps)

    This is a linear operation, so intervals are unchanged.

    Args:
        a: constant term (batch, features) - GPU tensor
        b: coefficient term (batch, features) - GPU tensor
        params: (gamma, beta, running_mean, running_var, eps)
                gamma/beta can be None if affine=False
                running_mean/var can be None if track_running_stats=False

    Returns:
        a_out, b_out (intervals unchanged - linear operation)
    """
    gamma, beta, running_mean, running_var, eps = params

    # Handle track_running_stats=False: compute from input batch
    if running_mean is None or running_var is None:
        running_mean = torch.mean(a, dim=0, keepdim=True)
        running_var = torch.var(a, dim=0, unbiased=False, keepdim=True)

    # Compute scale factor: 1 / sqrt(var + eps)
    std = torch.sqrt(running_var + eps)
    scale = 1.0 / std

    # Apply affine transform (gamma) if present
    if gamma is not None:
        scale = scale * gamma

    # Apply transformation: a_out = (a - mean) * scale + beta
    a_out = (a - running_mean) * scale
    if beta is not None:
        a_out = a_out + beta

    # Coefficient term: b_out = b * scale
    b_out = b * scale

    return a_out, b_out
