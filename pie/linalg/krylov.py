r"""
The Krylov subspace approximation is detailed in
`Analysis of some Krylov subspace approximations to the matrix exponential operator` from Y. Saad
in `SIAM Journal on Numerical Analysis, Vol. 29, No. 1`.
"""

import numpy as np
from scipy.linalg import expm as expm_sp


def expm_krylov(a, b, k, eps=1E-12):
    r"""
    Compute :math:`e^{a}\cdot b` using the Krylov subspace of dimension ``k``

    :param array_like a:
    :param array_like b:
    :param int k:
    :param eps: If the norm of the new computed Krylov basis vector is lower than ``eps``, it is assumed to be null
    :type eps: float, optional
    :return: numpy.ndarray
    """
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
        if norm < eps:
            break
        v_k[i] = v_new / norm

    h_k = np.dot(v_k, np.dot(a, v_k.T))
    return norm_0 * np.dot(v_k.T, expm_sp(h_k)[:, 0])
