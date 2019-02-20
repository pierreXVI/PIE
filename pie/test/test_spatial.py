import matplotlib.pyplot as plt
import numpy as np

import pie


def test_rhs(n, x_max, p, conv, diff, plot=False):
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
    ax1.grid(True)
    ax2.grid(True)
    ax1.plot(x, rhs_expected, 'k--', lw=3, label='Expected')
    ax2.plot(x, rhs_expected_burgers, 'k--', lw=3, label='Expected')

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


if __name__ == '__main__':
    test_rhs(n=100, x_max=1, p=5, conv=0.5, diff=0.1, plot=False)
    test_jac(n=100, x_max=1, p=5, conv=1.2, diff=0.0)
