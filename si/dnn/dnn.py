from .util import parse_model
from .layers import (
    apply_batchnorm1d_layer,
    apply_leaky_relu_layer,
    apply_linear_layer,
    apply_relu_layer,
)


def get_model_intervals(model, intervals, eps=1e-9):
    layers = parse_model(model)

    for layer_type, params in layers:
        # print(f"Processing layer: {layer_type}")
        if layer_type == "Linear":
            intervals = apply_linear_layer(intervals, params)

        elif layer_type == "ReLU":
            intervals = apply_relu_layer(intervals)

        elif layer_type == "LeakyReLU":
            intervals = apply_leaky_relu_layer(intervals, params)
        elif layer_type == "BatchNorm1d":
            intervals = apply_batchnorm1d_layer(intervals, params)

    intervals = sorted(intervals, key=lambda x: x[0])
    return intervals
