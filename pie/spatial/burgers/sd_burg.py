import numpy as np

from pie.spatial import sd
from pie.spatial.method import _SpatialMethod


class SpectralDifferenceMethodBurgers(_SpatialMethod):
    """
    Spatial scheme for viscous Burgers' equation,
    with a periodic boundary condition, using the spectral difference method.

    This method uses p + 1 flux points in the [-1, 1] cell, computed as the Legendre polynomial roots.

    :ivar array_like flux_pts: The repartition of the flux points inside a [-1, 1] cell
    :ivar array_like _sol_to_flux_conv_full: A needed matrix,
     ``_sol_to_flux_conv_full`` @ ``y`` is the flux expressed in the flux points multiplied by the scaling factor
    :ivar array_like _d_in_flux_to_sol_full: A needed matrix,
     ``_d_in_flux_to_sol_full`` @ ``f`` is the derivative of ``f`` expressed in the solution points,
     with ``f`` given in the flux points
    :ivar array_like _jac_diff: The constant jacobian for the diffusion part
    """

    def __init__(self, mesh, p, diff):
        super(SpectralDifferenceMethodBurgers, self).__init__(mesh, p, 0, diff)

        # Setting the solution points in a [-1, 1] cell as the Legendre roots
        self.flux_pts = np.append(-1, np.append(np.polynomial.legendre.legroots((self.p - 1) * [0] + [1]), 1))

        # Setting the needed matrices
        sol_to_flux = sd.lagrange_extrapolation_matrix(self.cell, self.flux_pts)
        flux_to_sol = sd.lagrange_extrapolation_matrix(self.flux_pts, self.cell)
        d_in_flux = np.zeros((self.p + 1, self.p + 1))
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                d_in_flux[i, j] = sd.d_lagrange(self.flux_pts[i], self.flux_pts, j)

        # Working with full size matrices
        isoparametric_scale = 2 / (np.roll(self.mesh, -1) - self.mesh)[:-1]
        self._sol_to_flux_conv_full = np.kron(np.diagflat(isoparametric_scale), sol_to_flux)
        sol_to_flux_diff_full = np.kron(np.diagflat(isoparametric_scale ** 2), sol_to_flux)
        d_in_flux_full = np.kron(np.eye(self.n_cell), d_in_flux)
        self._d_in_flux_to_sol_full = np.kron(np.eye(self.n_cell), np.dot(flux_to_sol, d_in_flux))

        # Continuity between cells
        riemann_diff = np.eye(self.n_cell * (self.p + 1))
        for i in range(self.n_cell):
            riemann_diff[i * (self.p + 1), i * (self.p + 1)] = 0.5
            riemann_diff[i * (self.p + 1), i * (self.p + 1) - 1] = 0.5
            riemann_diff[i * (self.p + 1) - 1, i * (self.p + 1)] = 0.5
            riemann_diff[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0.5

        self._jac_diff = self.d * np.dot(self._d_in_flux_to_sol_full,
                                         np.dot(riemann_diff, np.dot(d_in_flux_full,
                                                                     np.dot(riemann_diff, sol_to_flux_diff_full))))

    def rhs(self, y, t):
        """
        # Setting the needed matrices
        sol_to_flux = sd.lagrange_extrapolation_matrix(self.cell, self.flux_pts)
        flux_to_sol = sd.lagrange_extrapolation_matrix(self.flux_pts, self.cell)
        d_in_flux = np.zeros((self.p + 1, self.p + 1))
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                d_in_flux[i, j] = sd.d_lagrange(self.flux_pts[i], self.flux_pts, j)
        d_in_flux_to_sol = np.dot(flux_to_sol, d_in_flux)

        flux_in_flux_point_conv = np.zeros((self.n_cell, self.p + 1))
        flux_in_flux_point_diff = np.zeros((self.n_cell, self.p + 1))
        rhs_in_sol_point = np.zeros((self.n_cell, self.p))

        # Getting solution in sol points then the flux in flux points
        for i in range(self.n_cell):
            s = slice(i * self.p, (i + 1) * self.p)
            scale = 2 / (self.mesh[i + 1] - self.mesh[i])
            flux_in_flux_point_conv[i] = -(np.dot(sol_to_flux, y[s] * y[s] * scale / 2))
            flux_in_flux_point_diff[i] = self.d * (np.dot(sol_to_flux, y[s] * scale * scale))
        # Ensuring the flux continuity
        for i in range(self.n_cell):
            if y[i * self.p] + y[i * self.p - 1] > 0:
                flux_in_flux_point_conv[i, 0] = flux_in_flux_point_conv[i - 1, -1]
            else:
                flux_in_flux_point_conv[i - 1, -1] = flux_in_flux_point_conv[i, 0]
            riemann_diff = (flux_in_flux_point_diff[i, 0] + flux_in_flux_point_diff[i - 1, -1]) / 2
            flux_in_flux_point_diff[i, 0] = riemann_diff
            flux_in_flux_point_diff[i - 1, -1] = riemann_diff
        # Getting the flux in flux points for diffusion
        for i in range(self.n_cell):
            flux_in_flux_point_diff[i] = np.dot(d_in_flux, flux_in_flux_point_diff[i])
        # Ensuring the flux continuity for diffusion
        for i in range(self.n_cell):
            riemann_diff = (flux_in_flux_point_diff[i, 0] + flux_in_flux_point_diff[i - 1, -1]) / 2
            flux_in_flux_point_diff[i, 0] = riemann_diff
            flux_in_flux_point_diff[i - 1, -1] = riemann_diff
        # Getting the rhs in sol points
        for i in range(self.n_cell):
            rhs_in_sol_point[i] = np.dot(d_in_flux_to_sol, flux_in_flux_point_conv[i] + flux_in_flux_point_diff[i])
        # return rhs_in_sol_point.reshape(y.shape)
        return rhs_in_sol_point.reshape(y.shape)
        """
        j = np.dot(self._d_in_flux_to_sol_full, np.dot(self._riemann_solver(y), self._sol_to_flux_conv_full))
        return np.dot(self._jac_diff, y) - np.dot(j, y * y) / 2

    def jac(self, y, t):
        j = np.dot(self._d_in_flux_to_sol_full, np.dot(self._riemann_solver(y), self._sol_to_flux_conv_full))
        return self._jac_diff - j * y[None, :]

    def _riemann_solver(self, y):
        """

        :param array_like y: The flux
        :return: numpy.ndarray - the continuity matrix expressed in the flux points
        """
        # TODO: u_L < 0 < u_R
        # Continuity between cells
        riemann_conv = np.eye(self.n_cell * (self.p + 1))
        for i in range(self.n_cell):
            if y[i * self.p] >= 0 >= y[i * self.p - 1]:
                print('HAHA')
                riemann_conv[i * (self.p + 1), i * (self.p + 1)] = 0
                riemann_conv[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
            elif y[i * self.p] + y[i * self.p - 1] > 0:
                riemann_conv[i * (self.p + 1), i * (self.p + 1)] = 0
                riemann_conv[i * (self.p + 1), i * (self.p + 1) - 1] = 1
            else:
                riemann_conv[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
                riemann_conv[i * (self.p + 1) - 1, i * (self.p + 1)] = 1
        return riemann_conv

    def __repr__(self):
        foo = "Spectral difference for Burger's equation " + super(SpectralDifferenceMethodBurgers, self).__repr__()

        foo += "\n\nFlux points [-1, 1] :"
        cell_str = '   '.join(map('{0:0.2f}'.format, self.flux_pts))
        cell = len(cell_str) * ['-']
        for i in self.flux_pts:
            cell[int((len(cell) - 1) * (1 + i) / 2)] = '|'
        foo += "\n{0}\n{1}".format(cell_str, ''.join(cell))
        return foo
