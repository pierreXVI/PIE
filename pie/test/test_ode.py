from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from pie.temporal import rk, bdf, exp_taylor, exp_rosenbrock


def compare_methods(pb, h, t_max):
    r"""
    Compare methods in ``METHODS``

    :param pie.temporal.Problem pb: The problem to solve
    :param float h: The time step
    :param float t_max: Solve on [0, t_max]
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    t = np.linspace(0, t_max, int(t_max / h))

    y_exact = pb.y(t)
    if y_exact.ndim > 1:
        y_exact = y_exact[0, :]
    ax1.plot(t, y_exact, c='k', label=r'$y_{exact}$')

    score = dict()
    for method in METHODS:
        y = method(pb.y(0), t, pb.f, jac=pb.jac, jac2=pb.jac2, df_dt=pb.df_dt, d2f_dtdu=pb.d2f_dtdu, d2f_dt2=pb.d2f_dt2)
        if y.ndim > 1:
            y = y[:, 0]
        color = ax1.plot(t, y, '+', label=method.__name__)[0].get_color()
        ax2.semilogy(t[1:], abs(y - y_exact)[1:], c=color)
        score[method.__name__] = np.sum(abs(y - y_exact))

    print('\nScoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend()
    plt.show()


class Problem:
    r"""
    Generic ODE problem

    :param func y: Solution of the ODE problem
    :param func f: Function with well shaped input and output
    :param jac: The Jacobian of f, must return an array
    :type jac: func or None, optional
    :param jac2: The second-order Jacobian of f, must return an array
    :type jac2: func or None, optional
    :param df_dt: The f partial derivative with respect to time
    :type df_dt: func or None, optional
    :param d2f_dt2: The f second-order partial derivative with respect to time
    :type d2f_dt2: func or None, optional
    :param d2f_dtdu: The f crossed partial derivative, must return an array
    :type d2f_dtdu: func or None, optional
    """

    def __init__(self, y, f, jac=None, jac2=None, df_dt=None, d2f_dtdu=None, d2f_dt2=None):
        self.y = y
        self.f = f
        self.jac = jac
        self.jac2 = jac2
        self.df_dt = df_dt
        self.d2f_dtdu = d2f_dtdu
        self.d2f_dt2 = d2f_dt2


pb_1 = Problem(
    y=lambda t: np.exp(t / 2 - np.sin(2 * t) / 4),
    f=lambda y, t: y * (np.sin(t) ** 2),
    jac=lambda y, t: np.array([np.sin(t) ** 2]),
    jac2=lambda y, t: np.array([0]),
    df_dt=lambda y, t: y * np.sin(2 * t),
    d2f_dtdu=lambda y, t: np.array([np.sin(2 * t)]),
    d2f_dt2=lambda y, t: y * 2 * np.cos(2 * t)
)
r"""
From the `Wikipedia article <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ on RK methods

.. math::
   \left\{\begin{aligned}
    \dot{y} &= y\sin\left(t\right)^2 \\
    y\left(0\right) &= 0
   \end{aligned}\right.
   \Rightarrow y\left(t\right)=\exp\left(\frac{t}{2} - \frac{\sin\left(2t\right)}{4}\right)
"""

pb_2 = Problem(
    y=lambda t: np.sin(t) * np.exp(np.cos(t)),
    f=lambda y, t: np.cos(t) * np.exp(np.cos(t)) - y * y * np.exp(-np.cos(t)),
    jac=lambda y, t: np.array([- 2 * y * np.exp(-np.cos(t))]),
    jac2=lambda y, t: np.array([- 2 * np.exp(-np.cos(t))]),
    df_dt=lambda y, t: -np.sin(t) * (np.exp(np.cos(t)) * (1 + np.cos(t)) + y * y * np.exp(-np.cos(t))),
    d2f_dtdu=lambda y, t: np.array([- 2 * y * np.sin(t) * np.exp(-np.cos(t))]),
    d2f_dt2=lambda y, t: (np.sin(t) * np.sin(t) * (2 + np.cos(t)) - np.cos(t) * (1 + np.cos(t))) * np.exp(np.cos(t)) - (
            np.sin(t) * np.sin(t) + np.cos(t)) * y * y * np.exp(-np.cos(t))
)
r"""
.. math::
   \left\{\begin{aligned}
    \dot{y} &= \cos\left(t\right) e^{\cos\left(t\right)} - y^2e^{-\cos\left(t\right)} \\
    y\left(0\right) &= 0
   \end{aligned}\right.
   \Rightarrow y\left(t\right) = \sin\left(t\right) e^{\cos\left(t\right)}
"""

pb_3 = Problem(
    y=lambda t: 2 * np.arctan(np.tan(0.5) * np.exp(t)),
    f=lambda y, t: np.sin(y),
    jac=lambda y, t: np.array([np.cos(y)]),
    jac2=lambda y, t: np.array([-np.sin(y)])
)
r"""
.. math::
   \left\{\begin{aligned}
    \dot{y} &= \sin\left(y\right) \\
    y\left(0\right) &= 1
   \end{aligned}\right.
   \Rightarrow y\left(t\right) = 2\arctan\left(\tan\left(\frac{1}{2}\right)e^{-t}\right)
"""

pb_4 = Problem(
    y=lambda t: 0.1 / (1 - 0.1 * t),
    f=lambda y, t: y * y,
    jac=lambda y, t: np.array([2 * y]),
    jac2=lambda y, t: np.array([2])
)
r"""
The solutions of this equation do diverge in a finite time.

.. math::
   \left\{\begin{aligned}
    \dot{y} &= y^2 \\
    y\left(0\right) &= 0.1
   \end{aligned}\right.
   \Rightarrow y\left(t\right) = \frac{0.1}{1 - 0.1t}
"""

pb_5 = Problem(
    y=lambda t: (0 - 2500 / 2501) * np.exp(-50 * t) + 50 * np.sin(t) / 2501 + 2500 * np.cos(t) / 2501,
    f=lambda y, t: -50 * (y - np.cos(t)),
    jac=lambda y, t: np.array([-50]),
    jac2=lambda y, t: np.array([0])
)
r"""
Stiff equation, from Hairer & Wanner, 'Solving Ordinary Differential Equations II', chapter IV.1

.. math::
   \left\{\begin{aligned}
    \dot{y} &= -50\left(y - \cos\left(t\right)\right) \\
    y\left(0\right) &= 0
   \end{aligned}\right.
   \Rightarrow y\left(t\right) =  \frac{50}{2501}\sin\left(t\right)+ \frac{2500}{2501}\cos\left(t\right)
   -\frac{2500}{2501}e^{-50t}
"""

pb2d_1 = Problem(
    y=lambda t: np.array([np.cos(t), -np.sin(t)]),
    f=lambda y, t: np.array([y[1], -y[0]]),
    jac=lambda y, t: np.array([[0, 1], [-1, 0]]),
    jac2=lambda y, t: np.zeros((2, 2, 2))
)
r"""
Harmonic problem :
:math:`\left\{\begin{aligned}\ddot{y} + y &= 0 \\
\left(y, \dot{y}\right)_{t = 0} &= \left(1, 0\right)\end{aligned}\right.`
with :math:`y = \cos\left(t\right)` as solution
"""

pb2d_2 = Problem(
    y=lambda t: np.array([np.sin(t) * np.exp(np.cos(t)), (np.cos(t) - np.sin(t) * np.sin(t)) * np.exp(np.cos(t))]),
    f=lambda y, t: np.array([y[1], -(1 + 2 * np.cos(t)) * y[0] - np.sin(t) * y[1]]),
    jac=lambda y, t: np.array([[0, 1], [-(1 + 2 * np.cos(t)), - np.sin(t)]]),
    jac2=lambda y, t: np.zeros((2, 2, 2))
)
r"""
Same problem as ``pb_2`` written in 2D

.. math::
   \left\{\begin{aligned}
    &\ddot{y} + \sin\left(t\right)\dot{y} + \left(1 + 2\cos\left(t\right)\right)y = 0\\
    &\left(y, \dot{y}\right)_{t = 0} = \left(0, e\right)
   \end{aligned}\right.
   
with :math:`y = \sin\left(t\right) e^{\cos\left(t\right)}` as solution
"""

METHODS = (
    rk.rk_1,
    # rk.rk_2,
    # rk.rk_4,
    # rk.rk_butcher(rk.A_RK4, rk.B_RK4),
    bdf.bdf_1,
    # bdf.bdf_2,
    # bdf.bdf_3,
    # bdf.bdf_4,
    # bdf.bdf_5,
    # bdf.bdf_6,
    exp_taylor.taylor_exp_1,
    # exp_taylor.taylor_exp_2,
    # exp_taylor.taylor_exp_3,
    exp_rosenbrock.rosen_exp_1,
    # exp_rosenbrock.rosen_exp_2,
    # exp_rosenbrock.rosen_exp_3,
)
"""The methods that are going to be tested"""

if __name__ == '__main__':
    # compare_methods(pb_1, t_max=10, h=0.1)
    # compare_methods(pb_2, t_max=30, h=0.2)
    # compare_methods(pb_3, t_max=30, h=0.1)
    # compare_methods(pb_4, t_max=9, h=1)
    compare_methods(pb_5, t_max=1.5, h=1.5 / 39)
    # compare_methods(pb2d_1, t_max=10 * np.pi, h=0.05)
    # compare_methods(pb2d_2, t_max=10, h=0.01)

    pass
