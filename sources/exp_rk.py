import numpy as np
import utils


def exp_euler(y0, t, f):
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0
    a = np.atleast_2d(utils.jacobian(f, y[0], t[0]))
    for i in range(n - 1):
        h = (t[i + 1] - t[i])
        y[i + 1] = y[i] + np.dot(h * utils.phi_1(h * a), f(y[i], t[i]))
    return y


def exp_rosen(y0, t, f):
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0

    for i in range(n - 1):
        a = np.atleast_2d(utils.jacobian(f, y[i], t[i]))
        # print(a - np.sin(t[i]) ** 2)
        h = (t[i + 1] - t[i])
        y[i + 1] = y[i] + np.dot(h * utils.phi_1(h * a), f(y[i], t[i]))
    return y
