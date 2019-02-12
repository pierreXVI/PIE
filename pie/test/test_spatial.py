import matplotlib.pyplot as plt
import numpy as np

import pie


def test_rhs(n, x_max, p, conv, diff, plot=False):
    mesh = np.linspace(0, x_max, n + 1)

    method_fd = pie.spatial.FiniteDifferenceMethod(mesh, p, conv=conv, diff=diff)
    method_sd = pie.spatial.SpectralDifferenceMethod(mesh, p, conv=conv, diff=diff)
    method_fd_burgers = pie.spatial.burgers.FiniteDifferenceMethodBurgers(mesh, p, diff=diff)
    method_sd_burgers = pie.spatial.burgers.SpectralDifferenceMethodBurgers(mesh, p, diff=diff)

    x = method_fd.x
    k = 2 * np.pi / x_max
    y0 = np.sin(k * x)
    rhs_expected = -conv * k * np.cos(k * x) - diff * k * k * np.sin(k * x)
    rhs_expected_burgers = -y0 * k * np.cos(k * x) - diff * k * k * np.sin(k * x)

    rhs_fd = method_fd.rhs(y0, 0)
    rhs_sd = method_sd.rhs(y0, 0)
    rhs_fd_burgers = method_fd_burgers.rhs(y0, 0)
    rhs_sd_burgers = method_sd_burgers.rhs(y0, 0)

    err_fd = abs(rhs_expected - rhs_fd)
    err_sd = abs(rhs_expected - rhs_sd)
    err_fd_burgers = abs(rhs_expected_burgers - rhs_fd_burgers)
    err_sd_burgers = abs(rhs_expected_burgers - rhs_sd_burgers)
    print('Error on FiniteDifferenceMethod.rhs :\t\t\t {0:0.1E} (mean), {1:0.1E} (max)'
          .format(np.mean(err_fd), np.max(err_fd)))
    print('Error on SpectralDifferenceMethod.rhs :\t\t\t {0:0.1E} (mean), {1:0.1E} (max)'
          .format(np.mean(err_sd), np.max(err_sd)))
    print('Error on FiniteDifferenceMethodBurgers.rhs :\t {0:0.1E} (mean), {1:0.1E} (max)'
          .format(np.mean(err_fd_burgers), np.max(err_fd_burgers)))
    print('Error on SpectralDifferenceMethodBurgers.rhs :\t {0:0.1E} (mean), {1:0.1E} (max)'
          .format(np.mean(err_sd_burgers), np.max(err_sd_burgers)))
    print()

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.grid(True)
        ax2.grid(True)
        ax1.plot(x, rhs_expected, 'k--', lw=3, label='Expected')
        ax2.plot(x, rhs_expected_burgers, 'k--', lw=3, label='Expected')
        ax1.plot(x, rhs_fd, '--', label='FiniteDifferenceMethod')
        ax2.plot(x, rhs_fd_burgers, '--', label='FiniteDifferenceMethodBurgers')
        ax1.plot(x, rhs_sd, '--', label='SpectralDifferenceMethod')
        ax2.plot(x, rhs_sd_burgers, '--', label='SpectralDifferenceMethodBurgers')
        ax1.legend()
        ax2.legend()
        plt.show()


def test_jac(n, x_max, p, conv, diff):
    mesh = np.linspace(0, x_max, n + 1)

    eps = 1E-5

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


if __name__ == '__main__':
    test_rhs(n=100, x_max=1, p=5, conv=0.5, diff=0.1, plot=False)
    test_jac(n=100, x_max=1, p=5, conv=1.2, diff=0.0)
