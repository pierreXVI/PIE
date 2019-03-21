from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import pie


def compare_methods(pb, h, t_max, t=None, fmt='+-', title=''):
    r"""
    Compare methods in ``METHODS``

    :param pie.temporal.Problem pb: The problem to solve
    :param float h: The time step
    :param float t_max: Solve on [0, t_max]
    :param t: The time array. Override h ant t_max if given
    :type t: array_like, optional
    :param fmt: The line formatter
    :type fmt: str, optional
    :param title: The figure title
    :type title: str, optional
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$', fontsize=16)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$t$', fontsize=16)
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$', fontsize=16)

    if t is None:
        t = np.arange(0, int(t_max / h) + 1) * h
        if t[-1] != t_max:
            t = np.append(t, t_max)
    ax1.set_xticks(t, minor=True)

    y_exact = pb.y(t)
    t_full = np.linspace(0, t[-1], 1000)
    y_full = pb.y(t_full)
    if y_exact.ndim > 1:
        y_exact = y_exact[0, :]
        y_full = y_full[0, :]
    ax1.plot(t_full, y_full, c='k', lw=5, label=r'$y_{exact}$')

    if '-' in fmt:
        plot_options = {'lw': 3}
    else:
        plot_options = {'ms': 10, 'mew': 3}

    score = dict()
    for method in METHODS:
        y = method(pb.y(0), t, pb.f, jac=pb.jac, hess=pb.hess, df_dt=pb.df_dt, d2f_dtdu=pb.d2f_dtdu, d2f_dt2=pb.d2f_dt2)
        if y.ndim > 1:
            y = y[:, 0]
        color = ax1.plot(t, y, fmt, label=method.__name__, **plot_options)[0].get_color()
        ax2.semilogy(t[1:], abs(y - y_exact)[1:], c=color, lw=3)
        score[method.__name__] = np.sum(abs(y - y_exact))

    print('\nScoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend(fontsize=12, loc='upper right')
    fig.suptitle(title, fontsize=16)
    plt.show()


METHODS = (
    pie.temporal.rk_1,
    # pie.temporal.rk_2,
    # pie.temporal.rk_4,
    # pie.temporal.rk.rk_butcher(pie.temporal.rk.A_RK4, pie.temporal.rk.B_RK4),
    pie.temporal.bdf_1,
    # pie.temporal.bdf_2,
    # pie.temporal.bdf_3,
    # pie.temporal.bdf_4,
    # pie.temporal.bdf_5,
    # pie.temporal.bdf_6,
    # pie.temporal.taylor_exp_1,
    # pie.temporal.taylor_exp_2,
    # pie.temporal.taylor_exp_3,
    # pie.temporal.rosen_exp_1,
    # pie.temporal.rosen_exp_2,
    # pie.temporal.rosen_exp_3,
)
"""The methods that are going to be tested in ``compare_methods``"""

if __name__ == '__main__':
    compare_methods(pie.test.ode_problem.pb_0, t_max=5 * 2.2, h=2.2, title='Implicit vs Explicit')
    # compare_methods(pie.test.ode_problem.pb_1, t_max=10, h=0.1)
    # compare_methods(pie.test.ode_problem.pb_2, t_max=20, h=0.1)
    # compare_methods(pie.test.ode_problem.pb_3, t_max=30, h=0.1)
    # compare_methods(pie.test.ode_problem.pb_4, t_max=0, h=0, t=np.flip(10 - np.geomspace(1, 10, 20)))
    # compare_methods(pie.test.ode_problem.pb_5, t_max=20 * 0.04, h=0.04)
    # compare_methods(pie.test.ode_problem.pb2d_1, t_max=5 * np.pi, h=0.1)
    # compare_methods(pie.test.ode_problem.pb2d_1, t_max=10 * np.pi, h=2, fmt='+')
    # compare_methods(pie.test.ode_problem.pb2d_2, t_max=10, h=0.01)

    pass
