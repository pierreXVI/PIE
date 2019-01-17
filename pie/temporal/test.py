from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from pie.temporal import rk, bdf, exp


def compare_methods(pb, h, t_max):
    r"""
    Compare methods in ``METHODS``

    :param tuple pb: The problem to solve : y' = pb[0](y, t)
     with pb[1](y, t) the jacobian of pb[0] and y = pb[1] as solution
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

    y_exact = pb[-1](x)
    ax1.plot(x, y_exact, c='k', label=r'$y_{exact}$')

    score = dict()
    for method in METHODS:
        y = method(pb[-1](0), x, pb[0], jac=pb[1], jac2=pb[2])
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

    :param tuple pb: The problem to solve : [y, y']' = pb[0]([y, y'], t)
     with pb[1]([y, y'], t) the jacobian of pb[0], y = pb[2] as solution and [y, y'](0) = pb[3]
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

    y_exact = pb[-2](x)
    ax1.plot(x, y_exact, '+-', c='k', label=r'$y_{exact}$')

    score = dict()
    for method in METHODS:
        y = method(pb[-1], x, pb[0], jac=pb[1], jac2=pb[2])
        color = ax1.plot(x, y[:, 0], '+', label=method.__name__)[0].get_color()
        ax2.semilogy(x, abs(y[:, 0] - y_exact), c=color)
        score[method.__name__] = np.sum(abs(y[:, 0] - y_exact))

    print('Scoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend()
    plt.show()


METHODS = (
    rk.rk_1,
    # rk.rk_2,
    rk.rk_4,
    # rk.rk_butcher(rk.A_RK4, rk.B_RK4),
    bdf.bdf_1,
    # bdf.bdf_2,
    # bdf.bdf_3,
    # bdf.bdf_4,
    # bdf.bdf_5,
    # bdf.bdf_6,
    exp.taylor_exp_1,
    exp.taylor_exp_2,
    exp.taylor_exp_3,
)
"""The methods that are going to be tested"""

pb_1 = (lambda y, t: y * (np.sin(t) ** 2),
        lambda y, t: np.array([np.sin(t) ** 2]),
        lambda y, t: np.array([0]),
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
        lambda y, t: np.array([- 2 * y * np.exp(-np.cos(t))]),
        lambda y, t: np.array([- 2 * np.exp(-np.cos(t))]),
        lambda t: np.sin(t) * np.exp(np.cos(t)))
r"""
.. math::
   \left\{\begin{aligned}
    f\left(y, t\right) &= \cos\left(t\right) e^{\cos\left(t\right)} - y^2e^{-\cos\left(t\right)} \\
    y\left(t\right) &= \sin\left(t\right) e^{\cos\left(t\right)}
   \end{aligned}\right.
"""

pb_3 = (lambda y, t: np.sin(y),
        lambda y, t: np.array([np.cos(y)]),
        lambda y, t: np.array([-np.sin(y)]),
        lambda t: 2 * np.arctan(np.tan(0.5) * np.exp(t)))
r"""
.. math::
   \left\{\begin{aligned}
    f\left(y, t\right) &= \sin\left(y\right) \\
    y\left(t\right) &= 2\arctan\left(\tan\left(\frac{1}{2}\right)e^{-t}\right)
   \end{aligned}\right.
"""

pb_4 = (lambda y, t: y * y,
        lambda y, t: np.array([2 * y]),
        lambda y, t: np.array([2]),
        lambda t: 0.1 / (1 - 0.1 * t))
r"""
The solutions of this equation do diverge in a finite time.

.. math::
   \left\{\begin{aligned}
    f\left(y, t\right) &= y^2 \\
    y\left(t\right) &= \frac{0.1}{1 - 0.1t}
   \end{aligned}\right.
"""

pb2d_1 = (lambda y, t: np.array([y[1], -y[0]]),
          lambda y, t: np.array([[0, 1], [-1, 0]]),
          lambda y, t: np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
          lambda t: np.cos(t),
          np.array([1, 0]))
r"""
Harmonic problem :
:math:`\left\{\begin{aligned}\ddot{y} + y &= 0 \\
\left(y, \dot{y}\right)_{t = 0} &= \left(1, 0\right)\end{aligned}\right.`
with :math:`y = \cos\left(t\right)` as solution

"""

pb2d_2 = (lambda y, t: np.array([y[1], -(1 + 2 * np.cos(t)) * y[0] - np.sin(t) * y[1]]),
          lambda y, t: np.array([[0, 1], [-(1 + 2 * np.cos(t)), - np.sin(t)]]),
          lambda y, t: np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
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

# a, b = 1, 3
# pb2d_3 = (
#     lambda y, t: np.array([1 - (b + 1) * y[0] + a * y[0] * y[0] * y[1], b * y[0] - a * y[0] * y[0] * y[1]]),
#     lambda y, t: np.array([[-(b + 1) + 2 * a * y[0] * y[1], a * y[0] * y[0]],
#                            [b - 2 * a * y[0] * y[1], - a * y[0] * y[0]]]),
#     lambda y, t: np.array([[[2 * a * y[1], 2 * a * y[0]], [2 * a * y[0], 0]],
#                            [[- 2 * a * y[1], - 2 * a * y[1]], [-2 * a * y[0], 0]]]),
#     lambda t: np.sin(t) * np.exp(np.cos(t)),
#     np.array([1.5, 3]))

if __name__ == '__main__':
    # compare_methods(pb_1, t_max=10, h=0.1)
    # compare_methods(pb_2, t_max=50, h=0.1)
    # compare_methods(pb_3, t_max=30, h=0.1)
    compare_methods(pb_4, t_max=9, h=1)
    # compare_methods_2d(pb2d_1, t_max=10 * np.pi, h=0.05)
    # compare_methods_2d(pb2d_2, t_max=10, h=0.01)

    pass
