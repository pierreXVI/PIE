import scipy as sp
import numpy as np
from pie.temporal.commons import Counter


def taylor_exp(y0, t, f, jac, p=2, verbose=True, **_):
    # TODO: take the non linearity into account (the W array in the Koos paper)
    # TODO: decide a standard method (should p be a kwarg ?)
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n, d = len(t), 1
        y = np.zeros((n,))
    if verbose is False:
        count = Counter('', 0)
    elif verbose is True:
        count = Counter('Taylor Exp', n)
    else:
        count = Counter(verbose, n)
    expanded_vector = np.zeros((d + p,))
    expanded_vector[-1] = 1
    expanded_matrix = np.zeros((d + p, d + p))
    # expanded_matrix[-p:, -p:] = np.eye(p, k=1)
    expanded_matrix[-p:-1, -p + 1:] = np.eye(p - 1)
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, :d] = jac(y[i], t[i])
        y[i + 1] = np.dot(sp.linalg.expm(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y


def taylor_exp_1(y0, t, f, jac, verbose=True, **_):
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
    w = np.zeros((d, 1))
    expanded_vector = np.zeros((d + 1,))
    expanded_vector[-1] = 1
    expanded_matrix = np.zeros((d + 1, d + 1))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        j = jac(y[i], t[i])
        w[:, -1] = f(y[i], t[i]) - np.dot(j, y[i])
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, :d] = j
        expanded_matrix[:d, -1:] = w
        y[i + 1] = np.dot(sp.linalg.expm(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y


def taylor_exp_2(y0, t, f, jac, verbose=True, **_):
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
    w = np.zeros((d, 2))
    expanded_vector = np.zeros((d + 2,))
    expanded_vector[-1] = 1
    expanded_matrix = np.zeros((d + 2, d + 2))
    # expanded_matrix[-2:, -2:] = np.eye(2, k=1)
    expanded_matrix[-2:-1, -1:] = 1
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        j = jac(y[i], t[i])
        w[:, -1] = f(y[i], t[i]) - np.dot(j, y[i])
        w[:, -2] = np.dot(jac(y[i], t[i]) - j, f(y[i], t[i]))
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, :d] = j
        expanded_matrix[:d, -2:] = w
        y[i + 1] = np.dot(sp.linalg.expm(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y


def taylor_exp_3(y0, t, f, jac, jac2, verbose=True, **_):
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
    w = np.zeros((d, 3))
    expanded_vector = np.zeros((d + 3,))
    expanded_vector[-1] = 1
    expanded_matrix = np.zeros((d + 3, d + 3))
    # expanded_matrix[-3:, -3:] = np.eye(3, k=1)
    expanded_matrix[-3:-1, -2:] = np.eye(2)
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        j = jac(y[i], t[i])
        w[:, -1] = f(y[i], t[i]) - np.dot(j, y[i])
        w[:, -2] = np.dot(jac(y[i], t[i]) - j, f(y[i], t[i]))
        w[:, -3] = np.dot(np.dot(jac2(y[i], t[i]), f(y[i], t[i])), f(y[i], t[i])) \
                   + np.dot(jac(y[i], t[i]) - j, np.dot(jac(y[i], t[i]), y[i]))
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, :d] = j
        expanded_matrix[:d, -3:] = w
        y[i + 1] = np.dot(sp.linalg.expm(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y
