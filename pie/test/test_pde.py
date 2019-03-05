import matplotlib.pyplot as plt
import numpy as np

import pie
import pie.plot.animation


def solve(n, x_max, p, conv, diff, dt, t_max, init_cond, spatial_method, temporal_method):
    # Spatial
    mesh = np.linspace(0, x_max, n + 1)
    method = spatial_method(mesh, p, conv, diff)

    # Initial condition
    x = method.x
    y0 = init_cond(x)

    # Temporal
    cfl = max(abs(conv), abs(diff) / method.dx[0]) * dt / method.dx[0]
    t = np.append(np.arange(0, t_max, dt), t_max)

    # Solving
    y = temporal_method(y0, t, method.rhs, jac=method.jac, jac2=method.jac2, verbose='{0} + {1} at CFL = {2:0.3f}'
                        .format(temporal_method.__name__, spatial_method.__name__, cfl))

    return method, t, y


def solve_burgers(n, x_max, p, diff, dt, t_max, init_cond, spatial_method_burgers, temporal_method):
    # Spatial
    mesh = np.linspace(0, x_max, n + 1)
    method = spatial_method_burgers(mesh, p, diff)

    # Initial condition
    x = method.x
    y0 = init_cond(x)

    # Temporal
    t = np.append(np.arange(0, t_max, dt), t_max)
    cfl = max(np.max(abs(y0)), abs(diff) / method.dx[0]) * dt / method.dx[0]

    # Solving
    y = temporal_method(y0, t, method.rhs, jac=method.jac, jac2=method.jac2, verbose='{0} + {1} at CFL = {2:0.3f}'
                        .format(temporal_method.__name__, spatial_method_burgers.__name__, cfl))

    return method, t, y


def compare(n, x_max, p, conv, diff, dt, t_max, temporal_method, interval=200, repeat=False):
    y0 = pie.test.initial_condition.sine(x_max)
    method, t, y_fd = solve(n=n, x_max=x_max, p=p, conv=conv, diff=diff, dt=dt, t_max=t_max, init_cond=y0,
                            spatial_method=pie.spatial.FiniteDifferenceMethod, temporal_method=temporal_method)
    method, t, y_sd = solve(n=n, x_max=x_max, p=p, conv=conv, diff=diff, dt=dt, t_max=t_max, init_cond=y0,
                            spatial_method=pie.spatial.SpectralDifferenceMethod, temporal_method=temporal_method)

    pie.plot.animation.Animation(t, method.x, [y_fd, y_sd], ['FiniteDifferenceMethod', 'SpectralDifferenceMethod'])


def compare_burgers(n, x_max, p, diff, dt, t_max, init_cond, temporal_method, interval=200, repeat=False):
    method, t, y_fd = solve_burgers(n=n, x_max=x_max, p=p, diff=diff, dt=dt, t_max=t_max,
                                    init_cond=init_cond, temporal_method=temporal_method,
                                    spatial_method_burgers=pie.spatial.burgers.FiniteDifferenceMethodBurgers)
    method, t, y_sd = solve_burgers(n=n, x_max=x_max, p=p, diff=diff, dt=dt, t_max=t_max,
                                    init_cond=init_cond, temporal_method=temporal_method,
                                    spatial_method_burgers=pie.spatial.burgers.SpectralDifferenceMethodBurgers)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xticks(method.mesh)

    line_fd, = ax.plot(method.x, len(method.x) * [None], '+-', animated=True, label='FiniteDifferenceMethod')
    line_sd, = ax.plot(method.x, len(method.x) * [None], '+-', animated=True, label='SpectralDifferenceMethod')
    ax.legend()

    ax.set_xlim(0, x_max)
    ax.set_ylim(-1.2, 1.2)

    def update(frame):
        line_fd.set_ydata(y_fd[frame])
        line_sd.set_ydata(y_sd[frame])
        return line_fd, line_sd

    _ = FuncAnimation(fig, update, frames=len(t), blit=True, interval=interval, repeat=repeat)
    plt.show()


def plot(method, t, y, temporal_method_name='temporal_method'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid(True)
    ax1.set_xlabel(r'$x$')
    ax1.set_title('t = {0} (n_iter = {1}, dt = {2:0.2E})'.format(t[-1], len(t), t[1] - t[0]))
    ax1.plot(method.x, y[0], label='t = 0', c='k', lw=3)
    ax1.plot(method.x, y[-1], label='{0} + {1}'.format(temporal_method_name, method.__class__.__name__))
    ax1.legend()

    plt.show()


if __name__ == '__main__':
    # plot(*solve(n=50, x_max=1, p=4, conv=1, diff=0, dt=1E-3, t_max=1,
    #             init_cond=pie.test.initial_condition.sine(1),
    #             spatial_method=pie.spatial.SpectralDifferenceMethod,
    #             temporal_method=pie.temporal.rk_2))
    #
    # plot(*solve_burgers(n=50, x_max=1, p=4, diff=0, dt=1E-3, t_max=0.1,
    #                     init_cond=pie.test.initial_condition.sine(1),
    #                     spatial_method_burgers=pie.spatial.burgers.SpectralDifferenceMethodBurgers,
    #                     temporal_method=pie.temporal.rk_2))
    compare(n=50, x_max=1, p=4, conv=1, diff=0, dt=1E-3, t_max=5, temporal_method=pie.temporal.rk_2)
