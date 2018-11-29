import matplotlib.pyplot as plt
import numpy as np
from temporal import rk
from temporal import bdf


def compare_methods(pb, h, t_max):
    r"""
    Compare methods in ``METHODS``

    :param tuple pb: The problem to solve : y' = pb[0](y, t) with y = pb[1] as solution
    :param float h: The time step
    :param float t_max: Solve on [0, t_max]
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    x = np.linspace(0, t_max, int(t_max / h))

    y_exact = pb[1](x)
    ax1.plot(x, y_exact, c='k', label=r'$y_{exact}$')

    score = dict()
    for method in METHODS:
        y = method(pb[1](0), x, pb[0])
        color = ax1.plot(x, y, '+', label=method.__name__)[0].get_color()
        ax2.semilogy(x, abs(y - y_exact), c=color)
        score[method.__name__] = np.sum(abs(y - y_exact))

    print('Scoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend()
    plt.show()


def compare_methods_2d(pb, h, t_max):
    r"""
    Compare methods in ``METHODS`` on a 2D problem

    :param tuple pb: The problem to solve : [y, y']' = pb[0]([y, y'], t) with y = pb[1] as solution and [y, y'](0) = pb[2]
    :param float h: The time step
    :param float t_max: Solve on [0, t_max]
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    x = np.linspace(0, t_max, int(t_max / h))

    y_exact = pb[1](x)
    ax1.plot(x, y_exact, '+-', c='k', label=r'$y_{exact}$')

    score = dict()
    for method in METHODS:
        y = method(pb[2], x, pb[0])
        color = ax1.plot(x, y[:, 0], '+', label=method.__name__)[0].get_color()
        ax2.semilogy(x, abs(y[:, 0] - y_exact), c=color)
        score[method.__name__] = np.sum(abs(y[:, 0] - y_exact))

    print('Scoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend()
    plt.show()


METHODS = (
    # rk.rk_1,
    rk.rk_2,
    rk.rk_4,
    bdf.bdf_1,
    bdf.bdf_2,
    bdf.bdf_3,
    bdf.bdf_4,
    bdf.bdf_5,
    bdf.bdf_6,
    # implex.euler_implex,
    # implex.implex_2
    # exp_rk.exp_euler,
    # exp_rk.exp_rosen
)
"""The methods that are going to be tested"""

pb_1 = (lambda y, t: y * (np.sin(t) ** 2),
        lambda t: np.exp(t / 2 - np.sin(2 * t) / 4))
r"""
From the `Wikipedia article <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ on RK methods

.. math::
   \left\{\begin{aligned}
    f\left(y, t\right) &= y\sin\left(t\right)^2 \\
    y\left(t\right) &= \exp\left(\frac{t}{2} - \frac{\sin\left(2t\right)}{4}\right)
   \end{aligned}\right.
"""

pb_2 = (lambda y, t: np.cos(t) * np.exp(np.cos(t)) - y * y * np.exp(-np.cos(t)),
        lambda t: np.sin(t) * np.exp(np.cos(t)))
r"""
.. math::
   \left\{\begin{aligned}
    f\left(y, t\right) &= \cos\left(t\right) e^{\cos\left(t\right)} - y^2e^{-\cos\left(t\right)} \\
    y\left(t\right) &= \sin\left(t\right) e^{\cos\left(t\right)}
   \end{aligned}\right.
"""

pb2d_1 = (lambda y, t: np.array([y[1], -y[0]]),
          lambda t: np.cos(t),
          np.array([1, 0]))
r"""
Harmonic problem :
:math:`\left\{\begin{aligned}\ddot{y} + y &= 0 \\\left(y, \dot{y}\right)_{t = 0} &= \left(1, 0\right)\end{aligned}\right.`
with :math:`y = \cos\left(t\right)` as solution

"""

pb2d_2 = (lambda y, t: np.array([y[1], -(1 + 2 * np.cos(t)) * y[0] - np.sin(t) * y[1]]),
          lambda t: np.sin(t) * np.exp(np.cos(t)),
          np.array([0, np.e]))
r"""
Same problem as ``pb_2`` written in 2D

.. math::
   \left\{\begin{aligned}
    &\ddot{y} = -\left(1 + 2\cos\left(t\right)\right)y - \sin\left(t\right)\dot{y} \\
    &\left(y, \dot{y}\right)_{t = 0} = \left(0, e\right)
   \end{aligned}\right.

with :math:`y = \cos\left(t\right)` as solution
"""

if __name__ == '__main__':
    # compare_methods(pb_1, t_max=10, h=0.1)
    compare_methods(pb_2, t_max=300, h=0.1)
    # compare_methods_2d(pb2d_1, t_max=10 * np.pi, h=0.05)
    # compare_methods_2d(pb2d_2, t_max=100, h=0.1)

    pass
