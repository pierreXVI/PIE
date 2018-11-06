import matplotlib.pyplot as plt
import numpy as np
import rk
import bdf


def stability_rk1(alpha, dt_0, n_0, nb_test=4):
    r"""
    Test rk.rk_1 method on
    :math:`\left\{\begin{aligned}\dot{y} &= \alpha y \\ y_0 &= 1\end{aligned}\right.`

    for different values of the time step : (``dt_0``, ``dt_0 * 1E-1``, ..., ``dt_0 * 1E-(nb_tests - 1)``)

    :param alpha: the linear coefficient
    :param dt_0: the larger time step
    :param n_0: the number of points computed with dt = dt_0
    :param nb_test: the number of times steps tested
    """
    if nb_test < 1:
        return
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    for k in range(nb_test):
        x = np.arange(n_0 * (10 ** k)) * dt_0 * (0.1 ** k)
        y = rk.rk_1(1, x, lambda foo, bar: alpha * foo)
        color = ax1.plot(x, y, '+', label=r'$\Delta t = {0:0.1E}$'.format(dt_0 * (0.1 ** k)))[0].get_color()
        y_exact = np.exp(alpha * x)
        ax2.semilogy(x, abs(y - y_exact), c=color)

    ax1.plot(x, y_exact, 'k--', label=r'$y_{exact}$')

    ax2.plot(x, np.exp(alpha * x), 'k-.', label=r'$\exp\left(\alpha t\right)$')
    ax1.legend()
    ax2.legend()
    plt.show()


def stability_bdf1(alpha, dt_0, n_0, nb_test=4):
    r"""
    Test bdf.bdf_1 method on
    :math:`\left\{\begin{aligned}\dot{y} &= \alpha y \\ y_0 &= 1\end{aligned}\right.`

    for different values of the time step : (``dt_0``, ``dt_0 * 1E-1``, ..., ``dt_0 * 1E-(nb_tests - 1)``)

    :param alpha: the linear coefficient
    :param dt_0: the larger time step
    :param n_0: the number of points computed with dt = dt_0
    :param nb_test: the number of times steps tested
    """
    if nb_test < 1:
        return
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    for k in range(nb_test):
        x = np.arange(n_0 * (10 ** k)) * dt_0 * (0.1 ** k)
        y = bdf.bdf_1(1, x, lambda foo, bar: alpha * foo)
        color = ax1.plot(x, y, '+', label=r'$\Delta t = {0:0.1E}$'.format(dt_0 * (0.1 ** k)))[0].get_color()
        y_exact = np.exp(alpha * x)
        ax2.semilogy(x, abs(y - y_exact), c=color)

    ax1.plot(x, y_exact, 'k--', label=r'$y_{exact}$')

    ax2.plot(x, np.exp(alpha * x), 'k-.', label=r'$\exp\left(\alpha t\right)$')
    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    stability_rk1(0.8, 0.2, 50, nb_test=5)
    stability_bdf1(0.8, 0.2, 50, nb_test=5)
