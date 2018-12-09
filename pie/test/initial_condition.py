import numpy as np


def gaussian():
    def y0(x):
        u1 = min(x)
        u2 = max(x)
        u = (x - u1) / (u2 - u1)
        return np.exp(-200. * (u - 0.5) ** 2)

    return y0


def sine(n_period=1):
    def y0(x):
        u1 = min(x)
        u2 = max(x)
        u = (x - u1) / (u2 - u1)
        return np.sin(2 * np.pi * u * n_period)

    return y0


def trimmed_sine(n_period=1):
    def y0(x):
        u1 = min(x)
        u2 = max(x)
        u = (x - u1) / (u2 - u1)
        return np.sin(4 * np.pi * u * n_period) * (u > 1 / 4) * (u < 3 / 4)

    return y0


def rect():
    def y0(x):
        u1 = min(x)
        u2 = max(x)
        u = (x - u1) / (u2 - u1)
        return np.ones(x.shape) * (u > 1 / 4) * (u < 3 / 4)

    return y0
