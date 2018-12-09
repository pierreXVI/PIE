import matplotlib.pyplot as plt
import numpy as np
from pie.test import initial_condition
from pie import temporal
from pie import spatial


def solve(n, x_max, order, c, cfl, n_period, init_cond, spatial_method, temporal_method):
    # Spatial
    mesh = np.linspace(0, x_max, n + 1)
    method = spatial_method(mesh, order, c)

    # Temporal
    dt = cfl * method.dx[0] / c
    t = np.append(np.arange(0, n_period * x_max / c, dt), n_period * x_max / c)

    # Initial condition
    x = method.x
    y0 = init_cond(x)

    # Solving
    y = temporal_method(y0, t, method.rhs)

    err = abs(y[-1] - y0)

    # Plotting
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_xlabel(r'$x$')
    ax1.set_title('Errors at t = {0} :\n'.format(t[-1]) + r'$L_1$ (mean) : {0:0.3E}'.format(np.mean(err))
                  + '\t' + r'$L_2$ : {0:0.3E}'.format(np.sqrt(np.mean(err ** 2)))
                  + '\t' + r'$L_\infty$ (max) : {0:0.3E}'.format(np.max(err)))
    ax1.plot(x, init_cond((x - c * t[-1]) % x_max), label='Expected', lw=3)
    ax1.plot(x, y[-1], label='{0} + {1}'.format(temporal_method.__name__, spatial_method.__name__))
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.grid(True)
    ax2.set_xlabel(r'$t$')
    err_1 = [np.mean(abs(y[i] - init_cond((method.x - c * t[i]) % x_max))) for i in range(len(t))]
    # err_inf = [np.max(abs(y[i] - initial_condition((method.x - c * t[i]) % x_max))) for i in range(len(t))]
    ax2.plot(t, err_1, '+', label=r'$\left\|\left\|y-y_{exact}\right\|\right\|_1\left(t\right)$')
    # ax2.plot(t, err_inf, '+', label=r'$\left\|\left\|y-y_{exact}\right\|\right\|_\infty\left(t\right)$')
    ax2.legend()

    plt.show()


def compare_spatial_order(n_min, x_max, order_max, c, dt, n_period, init_cond, spatial_method, temporal_method):
    # Temporal
    t = np.append(np.arange(0, n_period * x_max / abs(c), dt), n_period * x_max / abs(c))

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.grid(True)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.grid(True)
    ax3 = fig.add_subplot(313)
    ax3.grid(True)

    for p in range(1, order_max):
        # Spatial
        n = int(n_min * (order_max + 1) / (p + 1))
        mesh = np.linspace(0, x_max, n + 1)
        method = spatial_method(mesh, p, c)

        # Initial condition
        y0 = init_cond(method.x)

        # Solving
        y = temporal_method(y0, t, method.rhs, verbose='{0}, order {1}'.format(temporal_method.__name__, p + 1))
        err = abs(y[-1] - y0)
        cfl = abs(c) * dt / method.dx[0]
        ax1.plot(method.x, y[-1], label=r'Order {0} (CFL = {1:0.3f}, $n_{{cell}}$ = {2})'.format(p, cfl, method.n_cell))
        ax2.semilogy(method.x, abs(err))

        err_1 = [np.mean(abs(y[i] - init_cond((method.x - c * t[i]) % x_max))) for i in range(len(t))]
        # err_inf = [np.max(abs(y[i] - initial_condition((method.x - c * t[i]) % x_max))) for i in range(len(t))]
        ax3.plot(t, err_1, '+')
        # ax3.plot(t, err_inf, '+', label=r'$\left\|\left\|y-y_{exact}\right\|\right\|_\infty\left(t\right)$')
        ax3.set_title(r'$\left\|\left\|y-y_{exact}\right\|\right\|_1\left(t\right)$')

    ax1.legend()
    plt.show()


if __name__ == '__main__':
    solve(n=50, x_max=1, order=2, c=1, cfl=1, n_period=1,
          init_cond=initial_condition.sine(2),
          spatial_method=spatial.FiniteDifferenceMethod, temporal_method=temporal.rk_1)
    # compare_spatial_order(n_min=20, x_max=1, order_max=5, c=1, dt=1E-3, n_period=2,
    #                       initial_condition=test.initial_condition.sine(2),
    #                       spatial_method=spatial.SpectralDifferenceMethod, temporal_method=temporal.rk_4)

    # mesh = np.linspace(0, 1, 10 + 1)
    # method = spatial.SpectralDifferenceMethod(mesh, 2, 1)
    # print(method.jac)
