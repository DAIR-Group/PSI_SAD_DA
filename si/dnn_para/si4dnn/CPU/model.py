from .operations import Linear, ReLU, LeakyReLU, BatchNorm1d
from . import util
import numpy as np


class CPUModel:
    def __init__(self, model):
        self.layers = util.parse_model(model)

    def forward(self, a, b, z):
        a = np.asarray(a)
        b = np.asarray(b)
        itv = np.array([-np.inf, np.inf])

        for name, params in self.layers:
            if name == "Linear":
                a, b = Linear(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z, itv)
            elif name == "LeakyReLU":
                # Extract alpha from params if available, otherwise use default
                alpha = params if params is not None else 0.01
                a, b, itv = LeakyReLU(a, b, z, itv, alpha)
            elif name == "BatchNorm1d":
                a, b = BatchNorm1d(a, b, params)
        return a, b, [itv[0], itv[1]]
