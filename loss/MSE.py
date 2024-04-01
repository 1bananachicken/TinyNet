import numpy as np
import nn.module as nn


class MeanSquaredError(nn.Module):
    def forward(self, y, t):
        return np.mean((y - t) ** 2)

    def backward(self, y, t):
        return 2 * (y - t)