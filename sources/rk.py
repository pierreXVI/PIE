import numpy as np


def rk_1(y0, t, f):
    """
    RK1 or Explicit Euler method
    y' = f(y, t)
    y(t[0]) = y0

    :param y0: array_like -
        Initial value, may be multi-dimensional of size d
    :param t: 1D_array -
        Array of time steps, of size n
    :param f: func -
        Function with well shaped input and output
    :return: numpy.ndarray -
        The solution, of shape (n, d)
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

    :param y0: array_like -
        Initial value, may be multi-dimensional of size d
    :param t: 1D_array -
        Array of time steps, of size n
    :param f: func -
        Function with well shaped input and output
    :return: numpy.ndarray -
        The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        y[i + 1] = y[i] + h * k2
    return y


def rk_4(y0, t, f):
    """
    RK4 method
    y' = f(y, t)
    y(t[0]) = y0

    :param y0: array_like -
        Initial value, may be multi-dimensional of size d
    :param t: 1D_array -
        Array of time steps, of size n
    :param f: func -
        Function with well shaped input and output
    :return: numpy.ndarray -
        The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        k3 = f(y[i] + h * k2 / 2, t[i] + h / 2)
        k4 = f(y[i] + h * k3, t[i] + h)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def rk_butcher(y0, t, f, a, b):
    """
    Generic RK method, using a Butcher tableau (*a*, *b*, *c*)

    The *c* array is deduced from the *a* and the *b* array so that the method is consistent:
    :math:`c_{i}=\sum _{k=0}^{i-1}a_{ik}`

    :param y0: array_like -
        Initial value, may be multi-dimensional of size d
    :param t: 1D_array -
        Array of time steps, of size n
    :param f: func -
        Function with well shaped input and output
    :param a: 2D_array -
        The *a* array of the Butcher tableau
    :param b: 2D_array -
        The *b* array of the Butcher tableau
    :return: numpy.ndarray -
        The solution, of shape (n, d)
    """
    q = a.shape[0]
    c = np.array([np.sum(a[i, :i]) for i in range(q)])
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        p = np.zeros(q)
        for j in range(q):
            p[j] = f(y[i] + h * np.sum(a[j, :j] * p[:j]), t[i] + h * c[j])
        y[i + 1] = y[i] + h * np.sum(b * p)
    return y


A_RK4 = np.array([[0., 0., 0, 0],
                  [.5, 0., 0, 0],
                  [.0, .5, 0, 0],
                  [.0, 0., 1, 0]])
"""The *a* array for the RK4 Butcher tableau"""

B_RK4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
"""The *b* array for the RK4 Butcher tableau"""
