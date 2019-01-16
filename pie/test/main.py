import matplotlib.pyplot as plt
import numpy as np
from pie.test import initial_condition, sd_error
from pie import temporal
from pie import spatial


def solve(n, x_max, p, conv, diff, cfl, t_max, init_cond, spatial_method, temporal_method):
    # Spatial
    mesh = np.linspace(0, x_max, n + 1)
    method = spatial_method(mesh, p, conv, diff)

    # Temporal
    dt = np.inf
    if conv != 0:
        dt = cfl * method.dx[0] / abs(conv)
    if diff != 0:
        dt = min(dt, cfl * method.dx[0] * method.dx[0] / abs(diff))

    t = np.append(np.arange(0, t_max, dt), t_max)

    # Initial condition
    x = method.x
    y0 = init_cond(x)

    # Solving
    y = temporal_method(y0, t, method.rhs)

    return method, t, y


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
    plot(*solve(n=50, x_max=1, p=4, conv=1, diff=0, cfl=1, t_max=8.8,
                init_cond=initial_condition.sine(1),
                spatial_method=spatial.SpectralDifferenceMethod, temporal_method=temporal.rk_4))
