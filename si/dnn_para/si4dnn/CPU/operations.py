import numpy as np


def Linear(a, b, params):
    """
    (a + b * z) * W + B
    = (a * W + B) + (b * W * z)
    """
    X = np.stack([a, b], axis=0)  # (2, n, d)
    Y = np.matmul(X, params[0])  # (2, n, h)
    # Add bias with broadcasting
    a = Y[0] + params[1]  # shape (n, h) + (1, h) → (n, h)
    b = Y[1]  # shape (n, h)
    return a, b


def ReLU(a, b, z, itv):
    """
    X = a + b * z
    itv = [-inf, inf]
    where X <= 0:
        itv = intersect with solve for z: a + bz <= 0
        a = 0
        b = 0
    where X > 0:
        itv = intersect with solve for z: a + bz > 0
    """
    # Compute X = a + b * z
    X = a + b * z

    # Create masks for different regions
    negative_mask = X <= 0
    positive_mask = X > 0

    # Handle intervals - compute bounds where a + b*z <= 0 and a + b*z > 0
    # For a + b*z <= 0: z <= -a/b (if b > 0), z >= -a/b (if b < 0)
    # For a + b*z > 0: z > -a/b (if b > 0), z < -a/b (if b < 0)

    # Avoid division by zero
    b_nonzero = np.abs(b) > 1e-12
    threshold = np.where(b_nonzero, -a / b, np.inf)

    # Update intervals based on the sign of b
    b_positive = b > 0
    b_negative = b < 0

    # For negative region (X <= 0): set a = 0, b = 0
    a_out = np.where(negative_mask, 0.0, a)
    b_out = np.where(negative_mask, 0.0, b)

    # For positive region (X > 0): keep original a, b
    # (already handled by the where condition above)

    # Update intervals
    itv_out = itv.copy()

    # Where X <= 0, intersect with z <= -a/b (if b > 0) or z >= -a/b (if b < 0)
    negative_and_b_pos = negative_mask & b_positive & b_nonzero
    negative_and_b_neg = negative_mask & b_negative & b_nonzero

    # For negative region with positive b: z <= -a/b, so update upper bound
    if np.any(negative_and_b_pos):
        valid_thresholds = threshold[negative_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    # For negative region with negative b: z >= -a/b, so update lower bound
    if np.any(negative_and_b_neg):
        valid_thresholds = threshold[negative_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    # Where X > 0, intersect with z > -a/b (if b > 0) or z < -a/b (if b < 0)
    positive_and_b_pos = positive_mask & b_positive & b_nonzero
    positive_and_b_neg = positive_mask & b_negative & b_nonzero

    # For positive region with positive b: z > -a/b, so update lower bound
    if np.any(positive_and_b_pos):
        valid_thresholds = threshold[positive_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    # For positive region with negative b: z < -a/b, so update upper bound
    if np.any(positive_and_b_neg):
        valid_thresholds = threshold[positive_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    if itv_out[0] > itv_out[1]:
        return a_out, b_out, np.asarray([np.nan, np.nan])  # Invalid interval

    return a_out, b_out, itv_out


def BatchNorm1d(a, b, params):
    """
    BatchNorm1d: y = (x - mean) / sqrt(var + eps) * gamma + beta

    For symbolic form x = a + b * z:
        y = (a - mean) / std * gamma + beta + (b / std * gamma) * z

    Where std = sqrt(var + eps)

    This is a linear operation, so intervals are unchanged.

    Args:
        a: constant term (batch, features)
        b: coefficient term (batch, features)
        params: (gamma, beta, running_mean, running_var, eps)
                gamma/beta can be None if affine=False
                running_mean/var can be None if track_running_stats=False

    Returns:
        a_out, b_out (intervals unchanged - linear operation)
    """
    gamma, beta, running_mean, running_var, eps = params

    # Handle track_running_stats=False: compute from input batch
    if running_mean is None or running_var is None:
        running_mean = np.mean(a, axis=0, keepdims=True)
        running_var = np.var(a, axis=0, keepdims=True)

    # Compute scale factor: 1 / sqrt(var + eps)
    std = np.sqrt(running_var + eps)
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


def LeakyReLU(a, b, z, itv, alpha=0.01):
    """
    LeakyReLU activation: f(x) = max(alpha*x, x)
    X = a + b * z

    Where X <= 0:
        itv = intersect with solve for z: a + bz <= 0
        a = alpha * a
        b = alpha * b
    Where X > 0:
        itv = intersect with solve for z: a + bz > 0
        (keep a, b unchanged)

    Args:
        a: constant term
        b: coefficient term
        z: symbolic variable value
        itv: current interval bounds [lower, upper]
        alpha: negative slope (default 0.01)
    """
    # Compute X = a + b * z
    X = a + b * z

    # Create masks for different regions
    negative_mask = X <= 0
    positive_mask = X > 0

    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError(f"LeakyReLU alpha must be non-negative, got {alpha}")

    # Avoid division by zero
    b_nonzero = np.abs(b) > 1e-12
    threshold = np.where(b_nonzero, -a / b, np.inf)

    # Update intervals based on the sign of b
    b_positive = b > 0
    b_negative = b < 0

    # For negative region (X <= 0): scale by alpha
    a_out = np.where(negative_mask, alpha * a, a)
    b_out = np.where(negative_mask, alpha * b, b)

    # Update intervals (same logic as ReLU since threshold points are identical)
    itv_out = itv.copy()

    # Where X <= 0, intersect with z <= -a/b (if b > 0) or z >= -a/b (if b < 0)
    negative_and_b_pos = negative_mask & b_positive & b_nonzero
    negative_and_b_neg = negative_mask & b_negative & b_nonzero

    # For negative region with positive b: z <= -a/b, so update upper bound
    if np.any(negative_and_b_pos):
        valid_thresholds = threshold[negative_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    # For negative region with negative b: z >= -a/b, so update lower bound
    if np.any(negative_and_b_neg):
        valid_thresholds = threshold[negative_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    # Where X > 0, intersect with z > -a/b (if b > 0) or z < -a/b (if b < 0)
    positive_and_b_pos = positive_mask & b_positive & b_nonzero
    positive_and_b_neg = positive_mask & b_negative & b_nonzero

    # For positive region with positive b: z > -a/b, so update lower bound
    if np.any(positive_and_b_pos):
        valid_thresholds = threshold[positive_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    # For positive region with negative b: z < -a/b, so update upper bound
    if np.any(positive_and_b_neg):
        valid_thresholds = threshold[positive_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    if itv_out[0] > itv_out[1]:
        return a_out, b_out, np.asarray([np.nan, np.nan])  # Invalid interval

    return a_out, b_out, itv_out
