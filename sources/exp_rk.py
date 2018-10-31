import numpy as np
import scipy.linalg


def exp_rk_1(y0, t, f):
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0

    for i in range(n - 1):
        a = np.atleast_2d(jacobian(f, y[i], t[i]))

        def g(u, t):
            return f(u, t) - np.dot(a, y[i])

        h = (t[i + 1] - t[i])
        y[i + 1] = np.dot(scipy.linalg.expm(h * a), y[i]) + h * g(y[i], t[i])

    return y


def rosenbrook(y0, t, f, a, b):
    c = np.array([np.sum(a[i, :i]) for i in range(a.shape[0])])
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[0] = y0

    for i in range(n - 1):
        a = np.atleast_2d(jacobian(f, y[i], t[i]))

        def g(u, t):
            return f(u, t) - np.dot(a, y[i])

        h = (t[i + 1] - t[i])
        y[i + 1] = np.dot(scipy.linalg.expm(h * a), y[i]) + h * g(y[i], t[i])

    return y


def one_step(u0, t0, t1, jac, g, a, b, c=None):
    if c is None:
        c = np.array([np.sum(a[i, :i]) for i in range(a.shape[0])])
    h = t1 - t0
    p = np.zeros(a.shape[0])
    for j in range(a.shape[0]):
        p[j] = f(u0 + h * np.sum(a[j, :j] * p[:j]), t0 + h * c[j])
    u1 = u0 + h * np.sum(b * p)
    return u1


def jacobian(f, u, t):
    eps = 1e-11
    try:
        d = len(u)
        jac = np.zeros((d, d))
        for j in range(d):
            e_j = np.zeros(d)
            e_j[j] = eps
            jac[:, j] = f(u + e_j, t) - f(u - e_j, t)
    except TypeError:
        jac = f(u + eps, t) - f(u - eps, t)
    return jac / (2 * eps)
