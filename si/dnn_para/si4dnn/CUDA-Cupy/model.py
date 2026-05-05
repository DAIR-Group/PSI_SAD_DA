from .operations import Linear, ReLU
from . import util
import cupy as cp
import numpy as np
import time

class CUDAModel:
    def __init__(self, model):
        self.layers = util.parse_model(model)

    def forward(self, a, b, z):
        a = cp.asarray(a, dtype=cp.float32)  # GPU array
        b = cp.asarray(b, dtype=cp.float32)  # GPU array
        itv = cp.array([-cp.inf, cp.inf], dtype=cp.float32)  # GPU interval
        z_gpu = cp.asarray(z, dtype=cp.float32)  # GPU scalar
        for name, params in self.layers:
            if name == "Linear":
                a, b = Linear(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z_gpu, itv)
        a = a.get()
        b = b.get()
        itv = [itv[0].get(), itv[1].get()]
        return a, b, itv