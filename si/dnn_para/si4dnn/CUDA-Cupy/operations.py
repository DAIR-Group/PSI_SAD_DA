import cupy as cp

def Linear(a, b, params): 
    stacked = cp.stack([a, b], axis=0)  # shape: (2, ...)
    result = cp.matmul(stacked, params[0])  # shape: (2, ...)
    result[0] += params[1]
    return result[0], result[1]

def relu_elementwise(a, b, z):
    X = a + b * z
    neg_mask = X <= 0
    b_nz = cp.abs(b) > 1e-12
    a_out = cp.where(neg_mask, 0.0, a)
    b_out = cp.where(neg_mask, 0.0, b)
    threshold = cp.where(b_nz, -a / b, cp.inf)
    where_min = (neg_mask & (b > 0)) | (~neg_mask & (b < 0))
    where_max = (neg_mask & (b < 0)) | (~neg_mask & (b > 0))
    return a_out, b_out, threshold, where_min, where_max

def ReLU(a, b, z, itv):
    a_out, b_out, threshold, where_min, where_max = relu_elementwise(a, b, z)
    min_val_array = cp.where(where_min, threshold, cp.inf)
    min_val = cp.min(min_val_array)
    max_val_array = cp.where(where_max, threshold, -cp.inf)
    max_val = cp.max(max_val_array)
    new_upper = cp.minimum(itv[1], min_val)
    new_lower = cp.maximum(itv[0], max_val)
    itv_out = cp.stack([new_lower, new_upper])
    itv_out = cp.where(new_lower <= new_upper, itv_out, cp.full_like(itv_out, cp.nan))
    return a_out, b_out, itv_out
