import torch


def apply_linear_layer(intervals, params):
    w, bias = params
    new_intervals = []
    for left, right, a_curr, b_curr in intervals:
        if w is not None:
            if a_curr.dtype != w.dtype:
                a_curr = a_curr.to(dtype=w.dtype)
            if b_curr.dtype != w.dtype:
                b_curr = b_curr.to(dtype=w.dtype)
        a_new = a_curr @ w
        if bias is not None:
            if bias.dtype != a_new.dtype:
                bias = bias.to(dtype=a_new.dtype)
            a_new = a_new + bias
        b_new = b_curr @ w
        new_intervals.append((left, right, a_new, b_new))
    return new_intervals


def apply_relu_layer(intervals):
    if len(intervals) == 0:
        return intervals

    new_intervals = []
    for left, right, a_curr, b_curr in intervals:
        with torch.no_grad():
            finite_mask = torch.abs(b_curr) >= 1e-12
            z_stars = torch.where(
                finite_mask,
                -a_curr / b_curr,
                torch.full_like(a_curr, float("inf")),
            )

            in_range = (z_stars > left) & (z_stars < right)
            interior = z_stars[in_range]

            if interior.numel() > 0:
                boundary = torch.tensor(
                    [left, right], dtype=interior.dtype, device=interior.device
                )
                splits = torch.cat([boundary, interior]).unique(sorted=True)
            else:
                splits = torch.tensor(
                    [left, right], dtype=a_curr.dtype, device=a_curr.device
                )

            sub_lefts = splits[:-1]
            sub_rights = splits[1:]
            z_mids = (sub_lefts + sub_rights) * 0.5

            pre_activations = a_curr.unsqueeze(0) + b_curr.unsqueeze(0) * z_mids.view(
                -1, 1, 1
            )
            active_masks = (pre_activations > 0).to(a_curr.dtype)

            a_news = a_curr.unsqueeze(0) * active_masks
            b_news = b_curr.unsqueeze(0) * active_masks

            for i in range(z_mids.shape[0]):
                new_intervals.append(
                    (
                        sub_lefts[i].item(),
                        sub_rights[i].item(),
                        a_news[i],
                        b_news[i],
                    )
                )

    return new_intervals


def apply_leaky_relu_layer(intervals, params):
    alpha = float(params)
    new_intervals = []

    for left, right, a_curr, b_curr in intervals:
        with torch.no_grad():
            finite_mask = torch.abs(b_curr) >= 1e-12
            z_stars = torch.where(
                finite_mask,
                -a_curr / b_curr,
                torch.full_like(a_curr, float("inf")),
            )

            in_range = (z_stars > left) & (z_stars < right)
            interior = z_stars[in_range]

            if interior.numel() > 0:
                boundary = torch.tensor(
                    [left, right], dtype=interior.dtype, device=interior.device
                )
                splits = torch.cat([boundary, interior]).unique(sorted=True)
            else:
                splits = torch.tensor(
                    [left, right], dtype=a_curr.dtype, device=a_curr.device
                )

            for i in range(splits.shape[0] - 1):
                sub_left = splits[i].item()
                sub_right = splits[i + 1].item()
                z_mid = 0.5 * (sub_left + sub_right)

                pre_activation = a_curr + b_curr * z_mid
                active_mask = (pre_activation > 0).to(a_curr.dtype)
                inactive_mask = 1.0 - active_mask

                a_new = a_curr * active_mask + alpha * a_curr * inactive_mask
                b_new = b_curr * active_mask + alpha * b_curr * inactive_mask

                new_intervals.append((sub_left, sub_right, a_new, b_new))

    return new_intervals


def apply_batchnorm1d_layer(intervals, params):
    gamma, beta, running_mean, running_var, bn_eps = params
    new_intervals = []

    if running_var is None or running_mean is None:
        raise ValueError(
            "BatchNorm1d in GPU path requires running_mean and running_var in eval mode."
        )

    scale = 1.0 / torch.sqrt(running_var + bn_eps)
    if gamma is not None:
        scale = gamma * scale

    shift = -running_mean * scale
    if beta is not None:
        shift = shift + beta

    for left, right, a_curr, b_curr in intervals:
        a_new = scale * a_curr + shift
        b_new = scale * b_curr
        new_intervals.append((left, right, a_new, b_new))

    return new_intervals
