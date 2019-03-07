import numpy as np

import pie
import pie.plot.animation


def solve(n, x_max, p, conv, diff, dt, t_max, init_cond, spatial_method, temporal_method, krylov_subspace_dim=None):
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
    y = temporal_method(y0, t, method.rhs, jac=method.jac, jac2=method.hess, krylov_subspace_dim=krylov_subspace_dim,
                        verbose='{0} + {1} at CFL = {2:0.3f}'
                        .format(temporal_method.__name__, spatial_method.__name__, cfl))

    return method, t, y


def solve_burgers(n, x_max, p, diff, dt, t_max, init_cond, spatial_method_burgers, temporal_method,
                  krylov_subspace_dim=None):
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
    y = temporal_method(y0, t, method.rhs, jac=method.jac, jac2=method.hess, krylov_subspace_dim=krylov_subspace_dim,
                        verbose='{0} + {1} at CFL = {2:0.3f}'
                        .format(temporal_method.__name__, spatial_method_burgers.__name__, cfl))

    return method, t, y


def compare(n, x_max, p, conv, diff, dt, t_max, krylov_subspace_dim=None, repeat=True, speed=1):
    y0 = pie.test.initial_condition.sine(x_max)
    list_y = []
    list_label = []
    list_fmt = []
    list_lw = []

    t, method = None, None
    for temporal_method in TEMPORAL_METHODS:
        for spatial_method in SPATIAL_METHODS:
            method, t, y = solve(n=n, x_max=x_max, p=p, conv=conv, diff=diff, dt=dt, t_max=t_max, init_cond=y0,
                                 spatial_method=spatial_method, temporal_method=temporal_method,
                                 krylov_subspace_dim=krylov_subspace_dim)
            list_y.append(y)
            if spatial_method == pie.spatial.FiniteDifferenceMethod:
                list_label.append('{0} + FD'.format(temporal_method.__name__))
            else:
                list_label.append('{0} + SD'.format(temporal_method.__name__))
            list_fmt.append('+-')
            list_lw.append(3)

    list_y = [[y0((method.x - conv * s) % x_max) * np.exp(-diff * s * ((2 * np.pi / x_max) ** 2)) for s in t]] + list_y
    list_label = ['Exact solution'] + list_label
    list_fmt = ['k'] + list_fmt
    list_lw = [5] + list_lw

    pie.plot.animation.Animation(t, method.x, list_y, list_label=list_label, list_fmt=list_fmt, list_lw=list_lw,
                                 x_ticks=np.linspace(0, x_max, n + 1), repeat=repeat, speed=speed)


def compare_burgers(n, x_max, p, diff, dt, t_max, init_cond, krylov_subspace_dim=None, repeat=True, speed=1):
    list_y = []
    list_label = []
    list_fmt = []
    list_lw = []

    t, method = None, None
    for temporal_method in TEMPORAL_METHODS:
        for spatial_method in SPATIAL_METHODS_BURGERS:
            method, t, y = solve_burgers(n=n, x_max=x_max, p=p, diff=diff, dt=dt, t_max=t_max, init_cond=init_cond,
                                         temporal_method=temporal_method, spatial_method_burgers=spatial_method,
                                         krylov_subspace_dim=krylov_subspace_dim)

            list_y.append(y)
            if spatial_method == pie.spatial.burgers.FiniteDifferenceMethodBurgers:
                list_label.append('{0} + FD Burgers'.format(temporal_method.__name__))
            else:
                list_label.append('{0} + SD Burgers'.format(temporal_method.__name__))
            list_fmt.append('+-')
            list_lw.append(1)

    pie.plot.animation.Animation(t, method.x, list_y, list_label=list_label, list_fmt=list_fmt, list_lw=list_lw,
                                 x_ticks=np.linspace(0, x_max, n + 1), repeat=repeat, speed=speed)


TEMPORAL_METHODS = (
    pie.temporal.rk_1,
    pie.temporal.rk_2,
    # pie.temporal.rk_4,
    pie.temporal.bdf_1,
    pie.temporal.bdf_2,
    # pie.temporal.bdf_4,
    pie.temporal.taylor_exp_1,
    pie.temporal.taylor_exp_2,
    # pie.temporal.taylor_exp_3,
    pie.temporal.rosen_exp_1,
    pie.temporal.rosen_exp_2,
    # pie.temporal.rosen_exp_3,
)
"""The temporal methods that are going to be tested"""

SPATIAL_METHODS = (
    # pie.spatial.FiniteDifferenceMethod,
    pie.spatial.SpectralDifferenceMethod,
)
"""The spatial methods that are going to be tested"""

SPATIAL_METHODS_BURGERS = (
    pie.spatial.burgers.FiniteDifferenceMethodBurgers,
    pie.spatial.burgers.SpectralDifferenceMethodBurgers,
)
"""The Burgers spatial methods that are going to be tested"""

if __name__ == '__main__':
    compare(n=30, x_max=1, p=3, conv=1, diff=0.0001, dt=1E-1, t_max=10,
            speed=1, repeat=False, krylov_subspace_dim=None)
    # compare(n=30, x_max=1, p=2, conv=1, diff=0.0001, dt=1E-2, t_max=10,
    #         speed=1, repeat=False, krylov_subspace_dim=None)
    # compare(n=20, x_max=1, p=3, conv=0.5, diff=0.00, dt=1E-1, t_max=50,
    #         speed=1, repeat=False, krylov_subspace_dim=None)
    # compare_burgers(n=127, x_max=1, p=4, diff=0.000, dt=1E-3, t_max=1, init_cond=pie.test.initial_condition.sine(1),
    #                 speed=1, repeat=False, krylov_subspace_dim=8)
    pass
