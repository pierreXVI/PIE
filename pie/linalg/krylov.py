import time

import numpy as np
from scipy.linalg import expm as expm_sp

import pie.temporal.commons


def expm_krylov(a, b, k, eps=1E-12):
    try:
        d = len(b)
    except TypeError:
        d = 1
    k = min(k, d)

    v_k = np.zeros((k, d))
    norm_0 = np.linalg.norm(b)
    v_k[0] = b / norm_0
    for i in range(1, k):
        v_new = np.dot(a, v_k[i - 1])
        for j in range(i):
            v_new = v_new - np.dot(v_new, v_k[j]) * v_k[j]
        norm = np.linalg.norm(v_new)
        if norm / norm_0 < eps:
            break
        v_k[i] = v_new / norm

    h_k = np.dot(v_k, np.dot(a, v_k.T))
    return np.dot(v_k.T, np.dot(expm_sp(h_k), np.dot(v_k, b)))


def test_expm_krylov(d, k, n):
    timer_sp = 0
    timer_kr = 0
    abs_err = []
    rel_err = []
    c = pie.temporal.commons.Counter('Test krylov vs scpy', n)
    for i in range(n):
        # a = np.random.rand(d, d)
        a = 100 * np.diagflat(np.random.rand(d))
        # a = np.diagflat(np.random.rand(d)) \
        #     + np.diagflat(np.random.rand(d - 1), k=1) \
        #     + np.diagflat(np.random.rand(d - 1), k=-1)
        b = np.random.rand(d)

        t = time.time()
        foo = np.dot(expm_sp(a), b)
        timer_sp += time.time() - t

        t = time.time()
        bar = expm_krylov(a, b, k)
        timer_kr += time.time() - t

        abs_err.append(abs(foo - bar))
        rel_err.append(abs((foo - bar) / foo))
        c(i)

    print("Mean elapsed time for {0} iterations : {1:0.1f} ms (scipy), {2:0.1f} ms (krylov)"
          .format(n, 1E3 * timer_sp / n, 1E3 * timer_kr / n))

    print('Relative error : {0:0.1E} (avg), {1:0.1E} (max)'
          .format(np.mean(rel_err), np.max(rel_err)))
    print('Absolute error : {0:0.1E} (avg), {1:0.1E} (max)'
          .format(np.mean(abs_err), np.max(abs_err)))


if __name__ == '__main__':
    test_expm_krylov(10, 5, 10)
