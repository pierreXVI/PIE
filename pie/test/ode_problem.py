r"""
Here are defined some Ordinary Differential Equations problems.
"""

import numpy as np


class Problem:
    r"""
    Generic ODE problem

    :param func y: Solution of the ODE problem
    :param func f: Function with well shaped input and output
    :param jac: The Jacobian of f, must return an array
    :type jac: func or None, optional
    :param hess: The second-order Jacobian of f, must return an array
    :type hess: func or None, optional
    :param df_dt: The f partial derivative with respect to time
    :type df_dt: func or None, optional
    :param d2f_dt2: The f second-order partial derivative with respect to time
    :type d2f_dt2: func or None, optional
    :param d2f_dtdu: The f crossed partial derivative, must return an array
    :type d2f_dtdu: func or None, optional
    """

    def __init__(self, y, f, jac=None, hess=None, df_dt=None, d2f_dtdu=None, d2f_dt2=None):
        self.y = y
        self.f = f
        self.jac = jac
        self.hess = hess
        self.df_dt = df_dt
        self.d2f_dtdu = d2f_dtdu
        self.d2f_dt2 = d2f_dt2


pb_0 = Problem(
    y=lambda t: np.exp(-t),
    f=lambda y, t: -y,
    jac=lambda y, t: np.array([1]),
)
r"""
Linear problem

.. math::
   \left\{\begin{aligned}
    &\dot{y} = -y \\
    &y\left(0\right) = 1
   \end{aligned}\right.
   \Rightarrow y\left(t\right)=\exp^{-t}
"""

pb_1 = Problem(
    y=lambda t: np.exp(t / 2 - np.sin(2 * t) / 4),
    f=lambda y, t: y * (np.sin(t) ** 2),
    jac=lambda y, t: np.array([np.sin(t) ** 2]),
    df_dt=lambda y, t: y * np.sin(2 * t),
    d2f_dtdu=lambda y, t: np.array([np.sin(2 * t)]),
    d2f_dt2=lambda y, t: y * 2 * np.cos(2 * t)
)
r"""
From the `Wikipedia article <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ on RK methods

.. math::
   \left\{\begin{aligned}
    &\dot{y} = y\sin\left(t\right)^2 \\
    &y\left(0\right) = 0
   \end{aligned}\right.
   \Rightarrow y\left(t\right)=\exp\left(\frac{t}{2} - \frac{\sin\left(2t\right)}{4}\right)
"""

pb_2 = Problem(
    y=lambda t: np.sin(t) * np.exp(np.cos(t)),
    f=lambda y, t: np.cos(t) * np.exp(np.cos(t)) - y * y * np.exp(-np.cos(t)),
    jac=lambda y, t: np.array([- 2 * y * np.exp(-np.cos(t))]),
    hess=lambda y, t: np.array([- 2 * np.exp(-np.cos(t))]),
    df_dt=lambda y, t: -np.sin(t) * (np.exp(np.cos(t)) * (1 + np.cos(t)) + y * y * np.exp(-np.cos(t))),
    d2f_dtdu=lambda y, t: np.array([- 2 * y * np.sin(t) * np.exp(-np.cos(t))]),
    d2f_dt2=lambda y, t: (np.sin(t) * np.sin(t) * (2 + np.cos(t)) - np.cos(t) * (1 + np.cos(t))) * np.exp(np.cos(t)) - (
            np.sin(t) * np.sin(t) + np.cos(t)) * y * y * np.exp(-np.cos(t))
)
r"""
.. math::
   \left\{\begin{aligned}
    &\dot{y} = \cos\left(t\right) e^{\cos\left(t\right)} - y^2e^{-\cos\left(t\right)} \\
    &y\left(0\right) = 0
   \end{aligned}\right.
   \Rightarrow y\left(t\right) = \sin\left(t\right) e^{\cos\left(t\right)}
"""

pb_3 = Problem(
    y=lambda t: 2 * np.arctan(np.tan(0.5) * np.exp(t)),
    f=lambda y, t: np.sin(y),
    jac=lambda y, t: np.array([np.cos(y)]),
    hess=lambda y, t: np.array([-np.sin(y)]),
)
r"""
.. math::
   \left\{\begin{aligned}
    &\dot{y} = \sin\left(y\right) \\
    &y\left(0\right) = 1
   \end{aligned}\right.
   \Rightarrow y\left(t\right) = 2\arctan\left(\tan\left(\frac{1}{2}\right)e^{-t}\right)
"""

pb_4 = Problem(
    y=lambda t: 0.1 / (1 - 0.1 * t),
    f=lambda y, t: y * y,
    jac=lambda y, t: np.array([2 * y]),
    hess=lambda y, t: np.array([2])
)
r"""
The solutions of this equation do diverge in a finite time.

.. math::
   \left\{\begin{aligned}
    &\dot{y} = y^2 \\
    &y\left(0\right) = 0.1
   \end{aligned}\right.
   \Rightarrow y\left(t\right) = \frac{0.1}{1 - 0.1t}
"""

pb_5 = Problem(
    y=lambda t: (0 - 2500 / 2501) * np.exp(-50 * t) + 50 * np.sin(t) / 2501 + 2500 * np.cos(t) / 2501,
    f=lambda y, t: -50 * (y - np.cos(t)),
    jac=lambda y, t: np.array([-50]),
    df_dt=lambda y, t: -50 * np.sin(t),
    d2f_dt2=lambda y, t: -50 * np.cos(t),
)
r"""
Stiff equation, from `Hairer & Wanner, 'Solving Ordinary Differential Equations II', chapter IV.1`

.. math::
   \left\{\begin{aligned}
    &\dot{y} = -50\left(y - \cos\left(t\right)\right) \\
    &y\left(0\right) = 0
   \end{aligned}\right.
   \Rightarrow y\left(t\right) =  \frac{50}{2501}\sin\left(t\right)+ \frac{2500}{2501}\cos\left(t\right)
   -\frac{2500}{2501}e^{-50t}
"""

pb2d_1 = Problem(
    y=lambda t: np.array([np.cos(t), -np.sin(t)]),
    f=lambda y, t: np.array([y[1], -y[0]]),
    jac=lambda y, t: np.array([[0, 1], [-1, 0]]),
)
r"""
Harmonic problem :
:math:`\left\{\begin{aligned}&\ddot{y} + y = 0 \\
&\left(y, \dot{y}\right)_{t = 0} = \left(1, 0\right)\end{aligned}\right.`
with :math:`y = \cos\left(t\right)` as solution
"""

pb2d_2 = Problem(
    y=lambda t: np.array([np.sin(t) * np.exp(np.cos(t)), (np.cos(t) - np.sin(t) * np.sin(t)) * np.exp(np.cos(t))]),
    f=lambda y, t: np.array([y[1], -(1 + 2 * np.cos(t)) * y[0] - np.sin(t) * y[1]]),
    jac=lambda y, t: np.array([[0, 1], [-(1 + 2 * np.cos(t)), - np.sin(t)]]),
    hess=lambda y, t: np.zeros((2, 2, 2)),
    df_dt=lambda y, t: np.array([0, 2 * np.sin(t) * y[0] - np.cos(t) * y[1]]),
    d2f_dtdu=lambda y, t: np.array([[0, 0], [2 * np.sin(t), - np.cos(t)]]),
    d2f_dt2=lambda y, t: np.array([0, 2 * np.cos(t) * y[0] + np.sin(t) * y[1]]),

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
