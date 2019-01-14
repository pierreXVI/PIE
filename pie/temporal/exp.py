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
    expanded_matrix[-p:, -p:] = np.eye(p, k=1)
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        expanded_vector[:d] = y[i]
        expanded_matrix[:d, :d] = jac(y[i], t[i])
        y[i + 1] = np.dot(sp.linalg.expm(h * expanded_matrix), expanded_vector)[:d]
        count(i + 1)
    return y
