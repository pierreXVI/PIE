from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import pie.test.ode_problem
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
        y = method(pb.y(0), t, pb.f, jac=pb.jac, hess=pb.hess, df_dt=pb.df_dt, d2f_dtdu=pb.d2f_dtdu, d2f_dt2=pb.d2f_dt2)
        if y.ndim > 1:
            y = y[:, 0]
        color = ax1.plot(t, y, '+', label=method.__name__)[0].get_color()
        ax2.semilogy(t[1:], abs(y - y_exact)[1:], c=color)
        score[method.__name__] = np.sum(abs(y - y_exact))

    print('\nScoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend()
    plt.show()


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
    # compare_methods(pie.test.ode_problem.pb_1, t_max=10, h=0.1)
    # compare_methods(pie.test.ode_problem.pb_2, t_max=30, h=0.2)
    # compare_methods(pie.test.ode_problem.pb_3, t_max=30, h=0.1)
    # compare_methods(pie.test.ode_problem.pb_4, t_max=9, h=1)
    compare_methods(pie.test.ode_problem.pb_5, t_max=1.5, h=1.5 / 39)
    # compare_methods(pie.test.ode_problem.pb2d_1, t_max=10 * np.pi, h=0.05)
    # compare_methods(pie.test.ode_problem.pb2d_2, t_max=10, h=0.01)

    pass
