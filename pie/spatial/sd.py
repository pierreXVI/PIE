import numpy as np
from pie.spatial.method import _SpatialMethod


# TODO: Write jac(self, y, t)
# TODO: Write the doc

class SpectralDifferenceMethod(_SpatialMethod):
    """

    """

    def __init__(self, mesh, p, conv):
        super(SpectralDifferenceMethod, self).__init__(mesh, p, conv)

        # Setting the solution points in a [-1, 1] cell as the Legendre roots
        self.flux_pts = np.append(-1, np.append(np.polynomial.legendre.legroots((self.p - 1) * [0] + [1]), 1))

        # Setting the needed matrices
        self.sol_to_flux = np.zeros((self.p + 1, self.p))
        self.flux_to_sol = np.zeros((self.p, self.p + 1))
        self.d_in_flux = np.zeros((self.p + 1, self.p + 1))
        for i in range(self.p):
            for j in range(self.p + 1):
                self.sol_to_flux[j, i] = lagrange(self.flux_pts[j], self.cell, i)
                self.flux_to_sol[i, j] = lagrange(self.cell[i], self.flux_pts, j)
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                self.d_in_flux[i, j] = d_lagrange(self.flux_pts[i], self.flux_pts, j)

        self.d_in_flux_to_sol = np.dot(self.flux_to_sol, self.d_in_flux)

    def rhs(self, y, t):
        flux_in_flux_point = np.zeros((self.n_cell, self.p + 1))
        rhs_in_sol_point = np.zeros((self.n_cell, self.p))

        # Getting solution in sol points then the flux in flux points
        for i in range(self.n_cell):
            sol_in_sol_point = y[i * self.p:(i + 1) * self.p] * 2 / (self.mesh[i + 1] - self.mesh[i])
            flux_in_flux_point[i] = -self.c * (np.dot(self.sol_to_flux, sol_in_sol_point))

        # Ensuring the flux continuity
        for i in range(self.n_cell):
            if self.c > 0:
                flux_in_flux_point[i, 0] = flux_in_flux_point[i - 1, -1]
            else:
                flux_in_flux_point[i - 1, -1] = flux_in_flux_point[i, 0]

        # Getting the rhs in sol points
        for i in range(self.n_cell):
            rhs_in_sol_point[i] = np.dot(self.d_in_flux_to_sol, flux_in_flux_point[i])

        return rhs_in_sol_point.reshape(y.shape)

    def jac(self, y, t):
        pass

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
