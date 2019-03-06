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


def compare(n, x_max, p, conv, diff, dt, t_max, temporal_method, repeat=True, speed=1):
    y0 = pie.test.initial_condition.sine(x_max)
    method, t, y_fd = solve(n=n, x_max=x_max, p=p, conv=conv, diff=diff, dt=dt, t_max=t_max, init_cond=y0,
                            spatial_method=pie.spatial.FiniteDifferenceMethod, temporal_method=temporal_method)
    method, t, y_sd = solve(n=n, x_max=x_max, p=p, conv=conv, diff=diff, dt=dt, t_max=t_max, init_cond=y0,
                            spatial_method=pie.spatial.SpectralDifferenceMethod, temporal_method=temporal_method)

    sol = [y0((method.x - conv * s) % x_max) * np.exp(-diff * s * ((2 * np.pi / x_max) ** 2)) for s in t]
    pie.plot.animation.Animation(t, method.x,
                                 [sol, y_fd, y_sd],
                                 list_label=['Exact solution', 'FiniteDifferenceMethod', 'SpectralDifferenceMethod'],
                                 list_fmt=['k', '+-', '+-'],
                                 list_lw=[3, 1, 1],
                                 repeat=repeat, speed=speed)


def compare_burgers(n, x_max, p, diff, dt, t_max, init_cond, temporal_method, repeat=True, speed=1):
    method, t, y_fd = solve_burgers(n=n, x_max=x_max, p=p, diff=diff, dt=dt, t_max=t_max,
                                    init_cond=init_cond, temporal_method=temporal_method,
                                    spatial_method_burgers=pie.spatial.burgers.FiniteDifferenceMethodBurgers)
    method, t, y_sd = solve_burgers(n=n, x_max=x_max, p=p, diff=diff, dt=dt, t_max=t_max,
                                    init_cond=init_cond, temporal_method=temporal_method,
                                    spatial_method_burgers=pie.spatial.burgers.SpectralDifferenceMethodBurgers)

    pie.plot.animation.Animation(t, method.x,
                                 [y_fd, y_sd],
                                 list_label=['FiniteDifferenceMethod', 'SpectralDifferenceMethod'],
                                 repeat=repeat, speed=speed)


if __name__ == '__main__':
    # compare(n=50, x_max=1, p=4, conv=1, diff=0.01, dt=1E-4, t_max=2, temporal_method=pie.temporal.rk_2,
    #         speed=10, repeat=True)
    compare_burgers(n=50, x_max=1, p=4, diff=0.00, dt=1E-3, t_max=1, init_cond=pie.test.initial_condition.rect(1),
                    temporal_method=pie.temporal.rk_4, speed=10, repeat=True)
