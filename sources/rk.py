import numpy as np


def rk_1(y0, t, f):
    """
    RK1 or Explicit Euler method
    y' = f(y, t)
    y(t[0]) = y0

    :param y0: initial value, may be multi-dimensional of size d
    :param t: array of time steps, of size n
    :param f: a function with well shaped input and output
    :return: the solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + (t[i + 1] - t[i]) * f(y[i], t[i])
    return y


def rk_2(y0, t, f):
    """
    RK2 or midpoint method
    y' = f(y, t)
    y(t[0]) = y0

    :param y0: initial value, may be multi-dimensional of size d
    :param t: array of time steps, of size n
    :param f: a function with well shaped input and output
    :return: the solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0
    for i in range(n - 1):
        h = (t[i + 1] - t[i])
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        y[i + 1] = y[i] + h * k2
    return y


def rk_4(y0, t, f):
    """
    RK4 method
    y' = f(y, t)
    y(t[0]) = y0

    :param y0: initial value, may be multi-dimensional of size d
    :param t: array of time steps, of size n
    :param f: a function with well shaped input and output
    :return: the solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0
    for i in range(n - 1):
        h = (t[i + 1] - t[i])
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        k3 = f(y[i] + h * k2 / 2, t[i] + h / 2)
        k4 = f(y[i] + h * k3, t[i] + h)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y
