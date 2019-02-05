import numpy as np


def lagrange(x, interpolation_points, i):
    r"""
    Evaluate in ``x`` the ``i`` Lagrange basis polynomial on the points ``interpolation_points``

    Returns :math:`L_i\left(x\right)`
    with :math:`L_i\left(interpolation\_points\left[j\right]\right) = \delta_{ij}`

    :param float x:
    :param array_like interpolation_points:
    :param int i:
    :return: float
    """
    foo = np.delete(interpolation_points, i)
    return np.prod((x - foo) / (interpolation_points[i] - foo))


def d_lagrange(x, interpolation_points, i):
    r"""
    Evaluate in ``x`` the derivative of the ``i`` Lagrange basis polynomial on the points ``interpolation_points``

    Returns :math:`\frac{\mathrm{d}L_i}{\mathrm{d}x}\left(x\right)`
    with :math:`L_i\left(interpolation\_points\left[j\right]\right) = \delta_{ij}`

    :param float x:
    :param array_like interpolation_points:
    :param int i:
    :return: float
    """
    val = 0
    foo = np.delete(interpolation_points, i)
    for k in range(len(foo)):
        val += np.prod(x - np.delete(foo, k))
    return val / np.prod(interpolation_points[i] - foo)


def lagrange_extrapolation_matrix(x, x_new):
    """
    Returns the change of basis matrix from the Lagrange interpolation polynomial on the points ``x``
    to the Lagrange interpolation polynomial on the points ``x_new``

    :param array_like x:
    :param array_like x_new:
    :return: array_like
    """
    a, b = len(x), len(x_new)
    foo = np.zeros((b, a))
    for i in range(a):
        for j in range(b):
            foo[j, i] = lagrange(x_new[j], x, i)
    return foo
