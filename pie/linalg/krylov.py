import numpy as np
from scipy.linalg import expm as expm_sp


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


if __name__ == '__main__':
    d = 1000
    k = 10

    # a = np.random.rand(d, d)
    a = np.diagflat(np.random.rand(d)) \
        + np.diagflat(np.random.rand(d - 1), k=1) \
        + np.diagflat(np.random.rand(d - 1), k=-1)
    b = np.random.rand(d)
    print(np.max(expm_sp(a)))
    import time

    t = time.time()
    foo = np.dot(expm_sp(a), b)
    print('Ellapsed scipy : {0:0.3f}'.format(time.time() - t))

    t = time.time()
    bar = expm_krylov(a, b, k)
    print('Ellapsed Krylov : {0:0.3f}'.format(time.time() - t))

    print('Relative error : {0:0.1%} (avg), {1:0.1%} (max)'
          .format(np.mean(abs(foo - bar) / foo), np.max(abs(foo - bar) / foo)))
    print('Absolute error : {0:0.1E} (avg), {1:0.1E} (max)'
          .format(np.mean(abs(foo - bar)), np.max(abs(foo - bar))))
