"""
Test
"""

import numpy as np


def gaussian(x_max):
    r"""

    :param float x_max:
    :return: :math:`\begin{array}{l|rcl}y_0 & \left[0, \textrm{x_max}\right] & \longrightarrow & \mathbb{R} \\
     &\textrm{x}&\longmapsto& \exp\left(-200\left(\frac{\textrm{x}}{\textrm{x_max}} - 0.5\right)^2\right)\end{array}`
    """

    def y0(x):
        return np.exp(-200. * (x / x_max - 0.5) ** 2)

    return y0


def sine(x_max, n_period=1):
    r"""

    :param float x_max:
    :param n_period:
    :type n_period: int, optional
    :return: :math:`\begin{array}{l|rcl}y_0 & \left[0, \textrm{x_max}\right] & \longrightarrow & \mathbb{R} \\
     & \textrm{x} & \longmapsto & \sin\left(2\pi\frac{\textrm{x}}{\textrm{x_max}}\times n\_period\right)\end{array}`
    """

    def y0(x):
        return np.sin(2 * np.pi * x / x_max * n_period)

    return y0


def trimmed_sine(x_max, n_period=1):
    r"""

    :param float x_max:
    :param n_period:
    :type n_period: int, optional
    :return: :math:`\begin{array}{l|rcl}y_0 & \left[0, \textrm{x_max}\right] & \longrightarrow & \mathbb{R} \\
     &\textrm{x}&\longmapsto&\begin{cases}\sin\left(4\pi\frac{\textrm{x}}{\textrm{x_max}}\times n\_period\right),
     &\text{if $\frac{1}{4}<\frac{\textrm{x}}{\textrm{x_max}}<\frac{3}{4}$} \\0,&\text{otherwise}\end{cases}\end{array}`
    """

    def y0(x):
        u = x / x_max
        return np.sin(4 * np.pi * u * n_period) * (u > 1 / 4) * (u < 3 / 4)

    return y0


def rect(x_max):
    r"""

    :param float x_max:
    :return: :math:`\begin{array}{l|rcl}y_0 & \left[0, \textrm{x_max}\right] & \longrightarrow & \mathbb{R} \\
     &\textrm{x}&\longmapsto&\begin{cases}1, &\text{if $\frac{1}{4}<\frac{\textrm{x}}{\textrm{x_max}}<\frac{3}{4}$} \\
     0, &\text{otherwise}\end{cases}\end{array}`
    """

    def y0(x):
        u = x / x_max
        return np.ones(x.shape) * (u > 1 / 4) * (u < 3 / 4)

    return y0
