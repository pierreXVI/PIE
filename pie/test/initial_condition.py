import numpy as np


def gaussian(x_max):
    def y0(x):
        return np.exp(-200. * (x / x_max - 0.5) ** 2)

    return y0


def sine(x_max, n_period=1):
    def y0(x):
        return np.sin(2 * np.pi * x / x_max * n_period)

    return y0


def trimmed_sine(x_max, n_period=1):
    def y0(x):
        u = x / x_max
        return np.sin(4 * np.pi * u * n_period) * (u > 1 / 4) * (u < 3 / 4)

    return y0


def rect(x_max):
    def y0(x):
        u = x / x_max
        return np.ones(x.shape) * (u > 1 / 4) * (u < 3 / 4)

    return y0
