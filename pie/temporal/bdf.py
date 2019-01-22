r"""
`BDF methods <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_ on Wikipedia.
"""

import warnings

import numpy as np
import scipy.optimize

from pie.temporal.commons import Counter
from pie.temporal.rk import rk_4


def _bdf_i(i, y0, t, f, func_to_minimise, jac_func_to_minimise, verbose):
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))

    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('BDF{0}'.format(i), n)
    else:
        count = Counter(verbose, n)

    y[:i] = rk_4(y0, t[:i], f, verbose=False)
    for k in range(n - i):
        result = scipy.optimize.root(func_to_minimise, y[k + i - 1], jac=jac_func_to_minimise,
                                     args=tuple([t[k + i - 1], t[k + i]] + [y[k + j] for j in range(i)]))
        if not result.success:
            warnings.warn(result.message)
        y[k + i] = result.x
        count(k + i)
    return y


def bdf_1(y0, t, f, verbose=True, jac=None, **_):
    """
    BDF1 or Implicit Euler method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param jac: If given, the Jacobian of f, must return an array
    :type jac: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """

    def func_to_minimise(u, t0, t1, u0):
        return u - u0 - (t1 - t0) * f(u, t1)

    if jac is None:
        jacobian = None
    else:
        def jacobian(u, t0, t1, *_):
            foo = jac(u, t1)
            return np.eye(*foo.shape) - (t1 - t0) * foo

    return _bdf_i(1, y0, t, f, func_to_minimise, jacobian, verbose)


def bdf_2(y0, t, f, verbose=True, jac=None, **_):
    """
    BDF2 method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param jac: If given, the Jacobian of f, must return an array
    :type jac: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """

    def func_to_minimise(u, t1, t2, u0, u1):
        return u - 4. * u1 / 3. + u0 / 3. - 2. * (t2 - t1) * f(u, t2) / 3.

    if jac is None:
        jacobian = None
    else:
        def jacobian(u, t1, t2, *_):
            foo = jac(u, t2)
            return np.eye(*foo.shape) - 2 * (t2 - t1) * foo / 3

    return _bdf_i(2, y0, t, f, func_to_minimise, jacobian, verbose)


def bdf_3(y0, t, f, verbose=True, jac=None, **_):
    """
    BDF3 method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param jac: If given, the Jacobian of f, must return an array
    :type jac: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """

    def func_to_minimise(u, t2, t3, u0, u1, u2):
        return u - 18. * u2 / 11. + 9. * u1 / 11. - 2. * u0 / 11. - 6 * (t3 - t2) * f(u, t3) / 11.

    if jac is None:
        jacobian = None
    else:
        def jacobian(u, t2, t3, *_):
            foo = jac(u, t3)
            return np.eye(*foo.shape) - 6 * (t3 - t2) * foo / 11

    return _bdf_i(3, y0, t, f, func_to_minimise, jacobian, verbose)


def bdf_4(y0, t, f, verbose=True, jac=None, **_):
    """
    BDF4 method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param jac: If given, the Jacobian of f, must return an array
    :type jac: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """

    def func_to_minimise(u, t3, t4, u0, u1, u2, u3):
        return u - 48. * u3 / 25. + 36. * u2 / 25. - 16. * u1 / 25. + 3. * u0 / 25. - 12 * (t4 - t3) * f(u, t4) / 25.

    if jac is None:
        jacobian = None
    else:
        def jacobian(u, t3, t4, *_):
            foo = jac(u, t4)
            return np.eye(*foo.shape) - 12 * (t4 - t3) * foo / 25

    return _bdf_i(4, y0, t, f, func_to_minimise, jacobian, verbose)


def bdf_5(y0, t, f, verbose=True, jac=None, **_):
    """
    BDF5 method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param jac: If given, the Jacobian of f, must return an array
    :type jac: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """

    def func_to_minimise(u, t4, t5, u0, u1, u2, u3, u4):
        return u - 300. * u4 / 137. + 300. * u3 / 137. - 200. * u2 / 137. + 75. * u1 / 137. - 12. * u0 / 137. \
               - 60 * (t5 - t4) * f(u, t5) / 137.

    if jac is None:
        jacobian = None
    else:
        def jacobian(u, t4, t5, *_):
            foo = jac(u, t5)
            return np.eye(*foo.shape) - 60 * (t5 - t4) * foo / 137

    return _bdf_i(5, y0, t, f, func_to_minimise, jacobian, verbose)


def bdf_6(y0, t, f, verbose=True, jac=None, **_):
    """
    BDF6 method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param jac: If given, the Jacobian of f, must return an array
    :type jac: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """

    def func_to_minimise(u, t5, t6, u0, u1, u2, u3, u4, u5):
        return u \
               - 360. * u5 / 147. \
               + 450. * u4 / 147. \
               - 400. * u3 / 147. \
               + 225. * u2 / 147. \
               - 72. * u1 / 147. \
               + 10. * u0 / 147. \
               - 60 * (t6 - t5) * f(u, t6) / 147.

    if jac is None:
        jacobian = None
    else:
        def jacobian(u, t5, t6, *_):
            foo = jac(u, t6)
            return np.eye(*foo.shape) - 60 * (t6 - t5) * foo / 147

    return _bdf_i(6, y0, t, f, func_to_minimise, jacobian, verbose)
