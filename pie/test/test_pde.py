import numpy as np

import pie
import pie.plot.animation


def solve(n, x_max, p, conv, diff, dt, t_max, init_cond_function, spatial_method, temporal_method,
          krylov_subspace_dim=None):
    """
    Solve a 1D-PDE

    :param int n: The number of mesh cells
    :param float x_max: The length of the spatial window (equation solved on [0, ``x_max``])
    :param int p: The number of points in a cell
    :param conv: If equals to 'burgers', solve the Burgers' equation, otherwise this is the convection parameter
    :type conv: int or 'burgers'
    :param float diff: The diffusion parameter
    :param float dt: The time step
    :param float t_max: The max time to solve (the last time step is trimmed to end on ``t_max``)
    :param func init_cond_function: The initial condition function
    :param pie.spatial._SpatialMethod spatial_method:
    :param pie.temporal._temporal_method temporal_method:
    :param krylov_subspace_dim: If given and the temporal method is exponential, uses the Krylov approximation
    :type krylov_subspace_dim: None or int, optional

    :return: (pie.spatial._SpatialMethod, numpy.ndarray, numpy.ndarray)
     - the spatial method, the time array and the solution for each computed time
    """

    # Spatial
    mesh = np.linspace(0, x_max, n + 1)
    if conv == 'burgers':
        method = spatial_method(mesh, p, diff)
    else:
        method = spatial_method(mesh, p, conv, diff)

    # Initial condition
    x = method.x
    y0 = init_cond_function(x)

    # Temporal
    if conv == 'burgers':
        cfl = max(np.max(abs(y0)), abs(diff) / method.dx[0]) * dt / method.dx[0]
    else:
        cfl = max(abs(conv), abs(diff) / method.dx[0]) * dt / method.dx[0]
    t = np.append(np.arange(0, t_max, dt), t_max)

    # Solving
    y = temporal_method(y0, t, method.rhs, jac=method.jac, jac2=method.hess, krylov_subspace_dim=krylov_subspace_dim,
                        verbose='{0} + {1} at CFL = {2:0.3f}'
                        .format(temporal_method.__name__, spatial_method.__name__, cfl))

    return method, t, y


def compare(n, x_max, p, conv, diff, dt, t_max, title='', krylov_subspace_dim=None, repeat=True, speed=1):
    """
    Compare spatial methods from ``SPATIAL_METHODS`` and temporal methods from ``TEMPORAL_METHODS``
    on an advection - diffusion PDE initialised with a sine and with periodic boundary conditions.
    Display the result on an animation.

    :param int n: The number of mesh cells
    :param float x_max: The length of the spatial window (equation solved on [0, ``x_max``])
    :param int p: The number of points in a cell
    :param conv: If equals to 'burgers', solve the Burgers' equation, otherwise this is the convection parameter
    :type conv: int or 'burgers'
    :param float diff: The diffusion parameter
    :param float dt: The time step
    :param float t_max: The max time to solve (the last time step is trimmed to end on ``t_max``)
    :param title: Title of the displayed figure
    :type title: str, optional
    :param krylov_subspace_dim: If given and the temporal method is exponential, uses the Krylov approximation
    :type krylov_subspace_dim: None or int, optional
    :param repeat: If True, the animation repeats itself when finished
    :type repeat: bool, optional
    :param speed: The animation speed at the beginning (can be changed)
    :type speed: float, optional
    """
    init_cond = pie.test.initial_condition.sine(x_max)
    list_y = []
    list_label = []
    list_fmt = []
    list_lw = []

    t, method = None, None
    for temporal_method in TEMPORAL_METHODS:
        for spatial_method in SPATIAL_METHODS:
            method, t, y = solve(n=n, x_max=x_max, p=p, conv=conv, diff=diff, dt=dt, t_max=t_max,
                                 init_cond_function=init_cond,
                                 spatial_method=spatial_method, temporal_method=temporal_method,
                                 krylov_subspace_dim=krylov_subspace_dim)
            list_y.append(y)
            if spatial_method == pie.spatial.FiniteDifferenceMethod:
                list_label.append('{0} + FD_1'.format(temporal_method.__name__))
            else:
                list_label.append('{0} + SD_{1}'.format(temporal_method.__name__, p - 1))
            list_fmt.append('+-')
            list_lw.append(3)

    list_y = [[init_cond((method.x - conv * s) % x_max) * np.exp(-diff * s * ((2 * np.pi / x_max) ** 2))
               for s in t]] + list_y
    list_label = ['Exact solution'] + list_label
    list_fmt = ['k'] + list_fmt
    list_lw = [5] + list_lw

    pie.plot.animation.Animation(t, method.x, list_y, list_label=list_label, list_fmt=list_fmt, list_lw=list_lw,
                                 x_ticks=np.linspace(0, x_max, n + 1), title=title, repeat=repeat, speed=speed)


def compare_burgers(n, x_max, p, diff, dt, t_max, init_cond, title='', krylov_subspace_dim=None, repeat=True, speed=1,
                    **init_cond_kwargs):
    """
    Compare spatial methods from ``SPATIAL_METHODS_BURGERS`` and temporal methods from ``TEMPORAL_METHODS``
    on a Burgers' PDE with periodic boundary conditions.
    Display the result on an animation.

    :param int n: The number of mesh cells
    :param float x_max: The length of the spatial window (equation solved on [0, ``x_max``])
    :param int p: The number of points in a cell
    :param float diff: The diffusion parameter
    :param float dt: The time step
    :param float t_max: The max time to solve (the last time step is trimmed to end on ``t_max``)
    :param func init_cond: To make the initial condition function
    :param title: Title of the displayed figure
    :type title: str, optional
    :param krylov_subspace_dim: If given and the temporal method is exponential, uses the Krylov approximation
    :type krylov_subspace_dim: None or int, optional
    :param repeat: If True, the animation repeats itself when finished
    :type repeat: bool, optional
    :param speed: The animation speed at the beginning (can be changed)
    :type speed: float, optional
    :param init_cond_kwargs: Optional keyword arguments used to make the initial condition
    """
    list_y = []
    list_label = []
    list_fmt = []
    list_lw = []

    t, method = None, None
    for temporal_method in TEMPORAL_METHODS:
        for spatial_method in SPATIAL_METHODS_BURGERS:
            method, t, y = solve(n=n, x_max=x_max, p=p, conv='burgers', diff=diff, dt=dt, t_max=t_max,
                                 init_cond_function=init_cond(x_max, **init_cond_kwargs),
                                 temporal_method=temporal_method, spatial_method=spatial_method,
                                 krylov_subspace_dim=krylov_subspace_dim)

            list_y.append(y)
            if spatial_method == pie.spatial.burgers.FiniteDifferenceMethodBurgers:
                list_label.append('{0} + FD_1 Burgers'.format(temporal_method.__name__))
            else:
                list_label.append('{0} + SD_{1} Burgers'.format(temporal_method.__name__, p - 1))
            list_fmt.append('+-')
            list_lw.append(3)

    pie.plot.animation.Animation(t, method.x, list_y, list_label=list_label, list_fmt=list_fmt, list_lw=list_lw,
                                 x_ticks=np.linspace(0, x_max, n + 1), title=title, repeat=repeat, speed=speed)


TEMPORAL_METHODS = (
    # pie.temporal.rk_1,
    # pie.temporal.rk_2,
    pie.temporal.rk_4,
    # pie.temporal.bdf_1,
    # pie.temporal.bdf_2,
    # pie.temporal.bdf_4,
    # pie.temporal.taylor_exp_1,
    # pie.temporal.taylor_exp_2,
    # pie.temporal.taylor_exp_3,
    # pie.temporal.rosen_exp_1,
    # pie.temporal.rosen_exp_2,
    # pie.temporal.rosen_exp_3,
)
"""The temporal methods that are going to be tested"""

SPATIAL_METHODS = (
    pie.spatial.FiniteDifferenceMethod,
    pie.spatial.SpectralDifferenceMethod,
)
"""The spatial methods that are going to be tested"""

SPATIAL_METHODS_BURGERS = (
    pie.spatial.burgers.FiniteDifferenceMethodBurgers,
    pie.spatial.burgers.SpectralDifferenceMethodBurgers,
)
"""The Burgers spatial methods that are going to be tested"""

if __name__ == '__main__':
    # Difference FD - SD
    # compare(n=50, x_max=1, p=2, conv=2, diff=0, dt=1E-3, t_max=1,
    #         speed=25, repeat=False, krylov_subspace_dim=None, title='FD vs SD')

    # Breaking bad
    # compare(n=30, x_max=1, p=3, conv=1, diff=0, dt=1E-2, t_max=0.5,
    #         speed=1, repeat=False, krylov_subspace_dim=None, title='Stabilité')

    # Advection 1
    # compare(n=20, x_max=1, p=3, conv=0.5, diff=0, dt=1E-2, t_max=100,
    #         speed=1, repeat=False, krylov_subspace_dim=None)

    # Advection - diffusion large CFL
    # compare(n=20, x_max=1, p=3, conv=0.5, diff=0.00005, dt=1E+1, t_max=100,
    #         speed=1, repeat=False, krylov_subspace_dim=None)

    # Burgers
    # compare_burgers(n=20, x_max=1, p=3, diff=0.000, dt=1E-3, t_max=1, init_cond=pie.test.initial_condition.sine,
    #                 speed=1, repeat=False, krylov_subspace_dim=None, n_period=3)

    pass
