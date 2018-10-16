import numpy as np
import scipy.optimize
from rk import rk_4


def bdf_1(y0, t, f):
    """
    BDF1 or Implicit Euler method
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
        def func_to_minimise(u):
            return u - y[i] - (t[i + 1] - t[i]) * f(u, t[i + 1])

        y[i + 1] = scipy.optimize.root(func_to_minimise, y[i]).x
    return y


def bdf_2(y0, t, f):
    """
    BDF2 method
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
    y[:2] = rk_4(y0, t[:2], f)
    for i in range(n - 2):
        def func_to_minimise(u):
            return u - 4. * y[i + 1] / 3. + y[i] / 3. - 2. * (t[i + 2] - t[i + 1]) * f(u, t[i + 2]) / 3.

        y[i + 2] = scipy.optimize.root(func_to_minimise, y[i + 1]).x
    return y


def bdf_3(y0, t, f):
    """
    BDF3 method
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
    y[:3] = rk_4(y0, t[:3], f)
    for i in range(n - 3):
        def func_to_minimise(u):
            return u \
                   - 18. * y[i + 2] / 11. \
                   + 9. * y[i + 1] / 11. \
                   - 2. * y[i] / 11. \
                   - 6 * (t[i + 3] - t[i + 2]) * f(u, t[i + 3]) / 11.

        y[i + 3] = scipy.optimize.root(func_to_minimise, y[i + 2]).x
    return y


def bdf_4(y0, t, f):
    """
    BDF4 method
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
    y[:4] = rk_4(y0, t[:4], f)
    for i in range(n - 4):
        def func_to_minimise(u):
            return u \
                   - 48. * y[i + 3] / 25. \
                   + 36. * y[i + 2] / 25. \
                   - 16. * y[i + 1] / 25. \
                   + 3. * y[i] / 25. \
                   - 12 * (t[i + 4] - t[i + 3]) * f(u, t[i + 4]) / 25.

        y[i + 4] = scipy.optimize.root(func_to_minimise, y[i + 3]).x
    return y


def bdf_5(y0, t, f):
    """
    BDF5 method
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
    y[:5] = rk_4(y0, t[:5], f)
    for i in range(n - 5):
        def func_to_minimise(u):
            return u \
                   - 300. * y[i + 4] / 137. \
                   + 300. * y[i + 3] / 137. \
                   - 200. * y[i + 2] / 137. \
                   + 75. * y[i + 1] / 137. \
                   - 12. * y[i] / 137. \
                   - 60 * (t[i + 5] - t[i + 4]) * f(u, t[i + 5]) / 137.

        y[i + 5] = scipy.optimize.root(func_to_minimise, y[i + 4]).x
    return y


def bdf_6(y0, t, f):
    """
    BDF6 method
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
    y[:6] = rk_4(y0, t[:6], f)
    for i in range(n - 6):
        def func_to_minimise(u):
            return u \
                   - 360. * y[i + 5] / 147. \
                   + 450. * y[i + 4] / 147. \
                   - 400. * y[i + 3] / 147. \
                   + 225. * y[i + 2] / 147. \
                   - 72. * y[i + 1] / 147. \
                   + 10. * y[i] / 147. \
                   - 60 * (t[i + 6] - t[i + 5]) * f(u, t[i + 6]) / 147.

        y[i + 6] = scipy.optimize.root(func_to_minimise, y[i + 5]).x
    return y


if __name__ == '__main__':
    pass
