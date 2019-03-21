r"""
This module is used to test the methods implemented in the ``spatial`` module.
"""


import matplotlib.pyplot as plt
import numpy as np

import pie


def test_rhs(n, x_max, p, conv, diff, plot=False):
    """
    Test the rhs methods of the spatial methods on a sine input and print the error.

    :param int n: The number of cells
    :param float x_max: The window size
    :param int p: The number of points inside a cell
    :param float conv: The convection parameter
    :param float diff: The diffusion parameter
    :param plot: If True, plot the computed RHS
    :type plot: bool, optional
    """
    mesh = np.linspace(0, x_max, n + 1)
    method_fd = pie.spatial.FiniteDifferenceMethod(mesh, p, conv=conv, diff=diff)
    x = method_fd.x
    k = 2 * np.pi / x_max
    y0 = np.sin(k * x)
    rhs_expected = -conv * k * np.cos(k * x) - diff * k * k * np.sin(k * x)
    rhs_expected_burgers = -y0 * k * np.cos(k * x) - diff * k * k * np.sin(k * x)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title('Convection - Diffusion')
    ax2.set_title('Viscous Burgers')
    ax1.grid(True)
    ax2.grid(True)
    ax1.plot(x, rhs_expected, 'k--', lw=3, label='Expected RHS')
    ax2.plot(x, rhs_expected_burgers, 'k--', lw=3, label='Expected RHS')

    for spatial_method in (pie.spatial.FiniteDifferenceMethod,
                           pie.spatial.SpectralDifferenceMethod):
        method = spatial_method(mesh, p, conv=conv, diff=diff)
        rhs = method.rhs(y0, 0)
        err = abs(rhs_expected - rhs)
        print('Error on {0}.rhs :\t\t\t {1:0.1E} (mean), {2:0.1E} (max)'
              .format(method.__class__.__name__, np.mean(err), np.max(err)))
        ax1.plot(x, rhs, '--', label=method.__class__.__name__)

    for spatial_method in (pie.spatial.burgers.FiniteDifferenceMethodBurgers,
                           pie.spatial.burgers.SpectralDifferenceMethodBurgers):
        method = spatial_method(mesh, p, diff=diff)
        rhs = method.rhs(y0, 0)
        err = abs(rhs_expected_burgers - rhs)
        print('Error on {0}.rhs :\t {1:0.1E} (mean), {2:0.1E} (max)'
              .format(method.__class__.__name__, np.mean(err), np.max(err)))
        ax2.plot(x, rhs, '--', label=method.__class__.__name__)

    print()

    ax1.legend()
    ax2.legend()
    if plot:
        plt.show()


def test_jac(n, x_max, p, conv, diff, eps=1E-5):
    r"""
    Test the jac methods of the spatial methods on a sine input and print the error.
    Uses an order 2 approximation of the jacobian as a reference value :
    :math:`\frac{\partial f}{\partial x}\left(x_0\right)
    \approx\frac{f\left(x_0+\varepsilon\right)-f\left(x_0-\varepsilon\right)}{2\varepsilon}`

    :param int n: The number of cells
    :param float x_max: The window size
    :param int p: The number of points inside a cell
    :param float conv: The convection parameter
    :param float diff: The diffusion parameter
    :param eps: Parameter used to approximate the jacobian
    :type eps: float, optional
    """
    mesh = np.linspace(0, x_max, n + 1)

    for spatial_method in (pie.spatial.FiniteDifferenceMethod,
                           pie.spatial.SpectralDifferenceMethod,
                           pie.spatial.burgers.FiniteDifferenceMethodBurgers,
                           pie.spatial.burgers.SpectralDifferenceMethodBurgers):
        try:
            method = spatial_method(mesh, p, conv=conv, diff=diff)
            msg = 'Difference between {0}.jac and order 2 approximation :\t\t\t {1:0.1E} (mean), {2:0.1E} (max)'
        except TypeError:
            method = spatial_method(mesh, p, diff=diff)
            msg = 'Difference between {0}.jac and order 2 approximation :\t {1:0.1E} (mean), {2:0.1E} (max)'

        y0 = np.sin(method.x * 2 * np.pi / x_max)

        jac = method.jac(y0, 0)
        jac_expected = np.zeros(jac.shape)
        for j in range(n * p):
            e_j = np.zeros(n * p)
            e_j[j] = eps
            jac_expected[:, j] = (method.rhs(y0 + e_j, 0) - method.rhs(y0 - e_j, 0)) / (2 * eps)
        err = abs(jac_expected - jac)
        print(msg.format(spatial_method.__name__, np.mean(err), np.max(err)))
    print()


def test_hess(n, x_max, p, conv, diff, eps=1E-5):
    r"""
    Test the hess methods of the spatial methods on a sine input and print the error.
    Uses an order 1 approximation of the hessian as a reference value :
    :math:`\frac{\partial^2 f}{\partial x^2}\left(x_0\right)
    \approx\frac{f\left(x_0+\varepsilon\right) - 2f\left(x_0\right) + f\left(x_0-\varepsilon\right)}{\varepsilon^2}`

    :param int n: The number of cells
    :param float x_max: The window size
    :param int p: The number of points inside a cell
    :param float conv: The convection parameter
    :param float diff: The diffusion parameter
    :param eps: Parameter used to approximate the hessian
    :type eps: float, optional
    """
    mesh = np.linspace(0, x_max, n + 1)

    for spatial_method in (pie.spatial.FiniteDifferenceMethod,
                           pie.spatial.SpectralDifferenceMethod,
                           pie.spatial.burgers.FiniteDifferenceMethodBurgers,
                           pie.spatial.burgers.SpectralDifferenceMethodBurgers):
        try:
            method = spatial_method(mesh, p, conv=conv, diff=diff)
            msg = 'Difference between {0}.hess and order 1 approximation :  \t\t {1:0.1E} (mean), {2:0.1E} (max)'
        except TypeError:
            method = spatial_method(mesh, p, diff=diff)
            msg = 'Difference between {0}.hess and order 1 approximation :\t {1:0.1E} (mean), {2:0.1E} (max)'

        y0 = np.sin(method.x * 2 * np.pi / x_max)

        hess = method.hess(y0, 0)
        hess_expected = np.zeros(hess.shape)
        for i in range(n * p):
            e_i = np.zeros(n * p)
            e_i[i] = eps
            for j in range(n * p):
                e_j = np.zeros(n * p)
                e_j[j] = eps
                hess_expected[:, i, j] = (method.rhs(y0 + e_i + e_j, 0) - method.rhs(y0 + e_i - e_j, 0)
                                          - method.rhs(y0 - e_i + e_j, 0) + method.rhs(y0 - e_i - e_j, 0)) \
                                         / (4 * eps * eps)
        err = abs(hess_expected - hess)
        print(msg.format(spatial_method.__name__, np.mean(err), np.max(err)))
    print()


if __name__ == '__main__':
    # test_rhs(n=100, x_max=1, p=5, conv=0.5, diff=0.1, plot=False)
    # test_jac(n=100, x_max=1, p=5, conv=1.2, diff=0.1)
    test_hess(n=20, x_max=1, p=3, conv=1.2, diff=0.1)
