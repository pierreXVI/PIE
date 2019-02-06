import numpy as np
from scipy.linalg import expm as expm_sp

from pie.temporal.commons import Counter


def rosen_exp_1(y0, t, f, jac, verbose=True, **_):
    """
    Order 1 Rosenbrock exponential method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param func jac: The Jacobian of f, must return an array
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n, d = len(t), 1
        y = np.zeros((n,))
    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('Taylor Exp 1', n)
    else:
        count = Counter(verbose, n)
    y[0] = y0
    w = np.zeros((d, 1))
    expanded_vector = np.zeros((d + 1,))
    expanded_vector[-1] = 1
    expanded_matrix = np.zeros((d + 1, d + 1))
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        j = jac(y[0], t[0])
        expanded_matrix[:d, :d] = j
        w[:, -1] = f(y[i], t[i]) - np.dot(j, y[i])
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, -1:] = w
        y[i + 1] = np.dot(expm_sp(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y


def rosen_exp_2(y0, t, f, jac, df_dt=None, verbose=True, **_):
    """
    Order 2 Rosenbrock exponential method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param func jac: The Jacobian of f, must return an array
    :param df_dt: The f partial derivative with respect to time
    :type df_dt: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n, d = len(t), 1
        y = np.zeros((n,))
    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('Taylor Exp 2', n)
    else:
        count = Counter(verbose, n)
    if df_dt is None:
        def df_dt(*_): return np.zeros((d,))
    y[0] = y0
    w = np.zeros((d, 2))
    expanded_vector = np.zeros((d + 2,))
    expanded_vector[-1] = 1
    expanded_matrix = np.zeros((d + 2, d + 2))
    expanded_matrix[-2:-1, -1:] = 1
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        j = jac(y[0], t[0])
        expanded_matrix[:d, :d] = j
        w[:, -1] = f(y[i], t[i]) - np.dot(j, y[i])
        w[:, -2] = np.dot(jac(y[i], t[i]) - j, f(y[i], t[i])) + df_dt(y[i], t[i])
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, -2:] = w
        y[i + 1] = np.dot(expm_sp(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y


def rosen_exp_3(y0, t, f, jac, jac2, df_dt=None, d2f_dt2=None, d2f_dtdu=None, verbose=True, **_):
    """
    Order 3 Rosenbrock exponential method

    :param array_like y0: Initial value, may be multi-dimensional of size d
    :param 1D_array t: Array of time steps, of size n
    :param func f: Function with well shaped input and output
    :param func jac: The Jacobian of f, must return an array
    :param func jac2: The second-order Jacobian of f, must return an array
    :param df_dt: The f partial derivative with respect to time
    :type df_dt: func or None, optional
    :param d2f_dt2: The f second-order partial derivative with respect to time
    :type d2f_dt2: func or None, optional
    :param d2f_dtdu: The f crossed partial derivative, must return an array
    :type d2f_dtdu: func or None, optional
    :param verbose: If True or a string, displays a progress bar
    :type verbose: bool or str, optional
    :return: numpy.ndarray - The solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n, d = len(t), 1
        y = np.zeros((n,))
    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('Taylor Exp 3', n)
    else:
        count = Counter(verbose, n)
    if df_dt is None:
        def df_dt(*_): return np.zeros((d,))
    if d2f_dt2 is None:
        def d2f_dt2(*_): return np.zeros((d,))
    if d2f_dtdu is None:
        def d2f_dtdu(*_): return np.zeros((d, d))
    y[0] = y0
    w = np.zeros((d, 3))
    expanded_vector = np.zeros((d + 3,))
    expanded_vector[-1] = 1
    expanded_matrix = np.zeros((d + 3, d + 3))
    expanded_matrix[-3:-1, -2:] = np.eye(2)
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        j = jac(y[0], t[0])
        expanded_matrix[:d, :d] = j
        w[:, -1] = f(y[i], t[i]) - np.dot(j, y[i])
        w[:, -2] = np.dot(jac(y[i], t[i]) - j, f(y[i], t[i])) + df_dt(y[i], t[i])
        w[:, -3] = np.dot(np.dot(jac2(y[i], t[i]), f(y[i], t[i])), f(y[i], t[i])) \
                   + np.dot(jac(y[i], t[i]) - j, np.dot(jac(y[i], t[i]), y[i])) \
                   + np.dot(jac(y[i], t[i]) - j, df_dt(y[i], t[i])) \
                   + 2 * np.dot(d2f_dtdu(y[i], t[i]), f(y[i], t[i])) \
                   + d2f_dt2(y[i], t[i])
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, -3:] = w
        y[i + 1] = np.dot(expm_sp(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y
