import numpy as np
from pie.temporal.commons import Counter


def rk_1(y0, t, f, verbose=True, **_):
    """
    RK1 or Explicit Euler method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('RK1', n)
    else:
        count = Counter(verbose, n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + (t[i + 1] - t[i]) * f(y[i], t[i])
        count(i + 1)
    return y


def rk_2(y0, t, f, verbose=True, **_):
    """
    RK2 or midpoint method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('RK2', n)
    else:
        count = Counter(verbose, n)
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        y[i + 1] = y[i] + h * k2
        count(i + 1)
    return y


def rk_4(y0, t, f, verbose=True, **_):
    """
    RK4 method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('RK4', n)
    else:
        count = Counter(verbose, n)
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        k3 = f(y[i] + h * k2 / 2, t[i] + h / 2)
        k4 = f(y[i] + h * k3, t[i] + h)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        count(i + 1)
    return y


def rk_butcher(a, b):
    """
    Generic RK method, using a Butcher tableau (*a*, *b*, *c*)

    The *c* array is deduced from the *a* and the *b* array so that the method is consistent:
    :math:`c_{i}=\sum _{k=0}^{i-1}a_{ik}`

    :param 2D_array a: The *a* array of the Butcher tableau
    :param 2D_array b: The *b* array of the Butcher tableau
    :return: func - the wanted Runge Kutta method
    """
    q = a.shape[0]
    c = np.array([np.sum(a[i, :i]) for i in range(q)])

    def rk_method(y0, t, f, verbose=True, **_):
        """
        RK method from given Butcher tableau

        :param array_like y0: Initial value, may be multi-dimensional of size d
        :param 1D_array t: Array of time steps, of size n
        :param func f: Function with well shaped input and output
        :param verbose: If True or a string, displays a progress bar
        :type verbose: bool or str, optional
        :return: numpy.ndarray - The solution, of shape (n, d)
        """
        try:
            n, d = len(t), len(y0)
            y = np.zeros((n, d))
            p = np.zeros((q, d))
        except TypeError:
            n = len(t)
            y = np.zeros((n,))
            p = np.zeros((q,))
        if verbose is False:
            count = Counter('', 0)
        elif verbose is True:
            count = Counter('RK_butcher', n)
        else:
            count = Counter(verbose, n)
        y[0] = y0
        for i in range(n - 1):
            h = t[i + 1] - t[i]
            for j in range(q):
                if p.ndim > 1:
                    p[j] = f(y[i] + h * np.sum(a[j, :j, None] * p[:j], axis=0), t[i] + h * c[j])
                else:
                    p[j] = f(y[i] + h * np.sum(a[j, :j] * p[:j]), t[i] + h * c[j])
            if p.ndim > 1:
                y[i + 1] = y[i] + h * np.sum(b[:, None] * p, axis=0)
            else:
                y[i + 1] = y[i] + h * np.sum(b * p)
            count(i + 1)
        return y

    return rk_method


A_RK4 = np.array([[0., 0., 0, 0],
                  [.5, 0., 0, 0],
                  [.0, .5, 0, 0],
                  [.0, 0., 1, 0]])
"""The *a* array for the RK4 Butcher tableau"""

B_RK4 = np.array([1. / 6, 1. / 3, 1. / 3, 1. / 6])
"""The *b* array for the RK4 Butcher tableau"""
