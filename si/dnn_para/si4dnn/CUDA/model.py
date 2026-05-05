from .operations import Linear, ReLU, LeakyReLU, BatchNorm1d
from . import util
import torch
import numpy as np
import time


class CUDAModel:
    def __init__(self, model):
        try:
            self.model_dtype = next(model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32
        self.layers = util.parse_model(model)

    def forward(self, a, b, z):
        a = torch.as_tensor(a, dtype=self.model_dtype, device="cuda")  # GPU tensor
        b = torch.as_tensor(b, dtype=self.model_dtype, device="cuda")  # GPU tensor
        itv = torch.tensor(
            [-float("inf"), float("inf")], dtype=self.model_dtype, device="cuda"
        )  # GPU interval
        z_gpu = torch.as_tensor(z, dtype=self.model_dtype, device="cuda")  # GPU scalar
        for name, params in self.layers:
            if name == "Linear":
                a, b = Linear(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z_gpu, itv)
            elif name == "LeakyReLU":
                # Extract alpha from params if available, otherwise use default
                alpha = params if params is not None else 0.01
                a, b, itv = LeakyReLU(a, b, z_gpu, itv, alpha)
            elif name == "BatchNorm1d":
                a, b = BatchNorm1d(a, b, params)
            # print(f"After layer {name}: a={a}, b={b}, itv={itv}")
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        itv = [itv[0].cpu().item(), itv[1].cpu().item()]
        return a, b, itv
