import numpy as np
from pie.spatial.method import _SpatialMethod


class SpectralDifferenceMethod(_SpatialMethod):
    """
    Spatial scheme for convection flux, with a periodic boundary condition, using the spectral difference method.

    This method uses p + 1 flux points in the [-1, 1] cell, computed as the Legendre polynomial roots.
    This method gives a linear right hand side so it has a constant jacobian, stored as a private attribute.

    :ivar array_like flux_pts: The repartition of the flux points inside a [-1, 1] cell
    :ivar array_like _jac: The constant jacobian
    """

    def __init__(self, mesh, p, conv, diff):
        super(SpectralDifferenceMethod, self).__init__(mesh, p, conv, diff)

        # Setting the solution points in a [-1, 1] cell as the Legendre roots
        self.flux_pts = np.append(-1, np.append(np.polynomial.legendre.legroots((self.p - 1) * [0] + [1]), 1))

        # Setting the needed matrices
        sol_to_flux = lagrange_extrapolation_matrix(self.cell, self.flux_pts)
        flux_to_sol = lagrange_extrapolation_matrix(self.flux_pts, self.cell)
        d_in_flux = np.zeros((self.p + 1, self.p + 1))
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                d_in_flux[i, j] = d_lagrange(self.flux_pts[i], self.flux_pts, j)

        # Working with full size matrices
        isoparametric_scale = 2 / (np.roll(self.mesh, -1) - self.mesh)[:-1]
        sol_to_flux_conv_full = np.kron(np.diagflat(isoparametric_scale), sol_to_flux)
        sol_to_flux_diff_full = np.kron(np.diagflat(isoparametric_scale ** 2), sol_to_flux)
        d_in_flux_full = np.kron(np.eye(self.n_cell), d_in_flux)
        d_in_flux_to_sol_full = np.kron(np.eye(self.n_cell), np.dot(flux_to_sol, d_in_flux))

        # Continuity between cells
        riemann_conv = np.eye(self.n_cell * (self.p + 1))
        riemann_diff = np.eye(self.n_cell * (self.p + 1))
        for i in range(self.n_cell):
            if self.c > 0:
                riemann_conv[i * (self.p + 1), i * (self.p + 1)] = 0
                riemann_conv[i * (self.p + 1), i * (self.p + 1) - 1] = 1
            else:
                riemann_conv[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
                riemann_conv[i * (self.p + 1) - 1, i * (self.p + 1)] = 1
            riemann_diff[i * (self.p + 1), i * (self.p + 1)] = 0.5
            riemann_diff[i * (self.p + 1), i * (self.p + 1) - 1] = 0.5
            riemann_diff[i * (self.p + 1) - 1, i * (self.p + 1)] = 0.5
            riemann_diff[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0.5

        self._jac = np.dot(d_in_flux_to_sol_full,
                           -self.c * np.dot(riemann_conv, sol_to_flux_conv_full)
                           + self.d * np.dot(riemann_diff, np.dot(d_in_flux_full,
                                                                  np.dot(riemann_diff, sol_to_flux_diff_full))))

    def rhs(self, y, t):
        return np.dot(self._jac, y)

    def jac(self, y, t):
        return self._jac

    def __repr__(self):
        foo = "Spectral difference " + super(SpectralDifferenceMethod, self).__repr__()

        foo += "\n\nFlux points [-1, 1] :"
        cell_str = '   '.join(map('{0:0.2f}'.format, self.flux_pts))
        cell = len(cell_str) * ['-']
        for i in self.flux_pts:
            cell[int((len(cell) - 1) * (1 + i) / 2)] = '|'
        foo += "\n{0}\n{1}".format(cell_str, ''.join(cell))
        return foo


def lagrange(x, interpolation_points, i):
    r"""
    Evaluate in ``x`` the ``i`` Lagrange basis polynomial on the points ``interpolation_points``

    Returns :math:`L_i\left(x\right)`
    with :math:`L_i\left(interpolation\_points\left[j\right]\right) = \delta_{ij}`

    :param float x:
    :param array_like interpolation_points:
    :param int i:
    :return: float
    """
    foo = np.delete(interpolation_points, i)
    return np.prod((x - foo) / (interpolation_points[i] - foo))


def d_lagrange(x, interpolation_points, i):
    r"""
    Evaluate in ``x`` the derivative of the ``i`` Lagrange basis polynomial on the points ``interpolation_points``

    Returns :math:`\frac{\mathrm{d}L_i}{\mathrm{d}x}\left(x\right)`
    with :math:`L_i\left(interpolation\_points\left[j\right]\right) = \delta_{ij}`

    :param float x:
    :param array_like interpolation_points:
    :param int i:
    :return: float
    """
    val = 0
    foo = np.delete(interpolation_points, i)
    for k in range(len(foo)):
        val += np.prod(x - np.delete(foo, k))
    return val / np.prod(interpolation_points[i] - foo)


def lagrange_extrapolation_matrix(x, x_new):
    """
    Returns the change of basis matrix from the Lagrange interpolation polynomial on the points x
    to the Lagrange interpolation polynomial on the points x_new

    :param array_like x:
    :param array_like x_new:
    :return: array_like
    """
    a, b = len(x), len(x_new)
    foo = np.zeros((b, a))
    for i in range(a):
        for j in range(b):
            foo[j, i] = lagrange(x_new[j], x, i)
    return foo
