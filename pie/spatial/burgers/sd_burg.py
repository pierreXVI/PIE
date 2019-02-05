import numpy as np

from pie.linalg import lagrange
from pie.spatial.method import _SpatialMethod


class SpectralDifferenceMethodBurgers(_SpatialMethod):
    """
    Spatial scheme for viscous Burgers' equation,
    with a periodic boundary condition, using the spectral difference method.

    This method uses p + 1 flux points in the [-1, 1] cell, computed as the Legendre polynomial roots.

    :ivar array_like flux_pts: The repartition of the flux points inside a [-1, 1] cell
    :ivar array_like _sol_to_flux_conv_full: A needed matrix,
     ``_sol_to_flux_conv_full`` @ ``y`` is the flux expressed in the flux points
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
        sol_to_flux = lagrange.lagrange_extrapolation_matrix(self.cell, self.flux_pts)
        flux_to_sol = lagrange.lagrange_extrapolation_matrix(self.flux_pts, self.cell)
        d_in_flux = np.zeros((self.p + 1, self.p + 1))
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                d_in_flux[i, j] = lagrange.d_lagrange(self.flux_pts[i], self.flux_pts, j)

        # Working with full size matrices
        isoparametric_scale = 2 / (np.roll(self.mesh, -1) - self.mesh)[:-1]
        self._sol_to_flux_full = np.kron(np.eye(self.n_cell), sol_to_flux)
        d_in_flux_full = np.kron(np.diagflat(isoparametric_scale), d_in_flux)
        self._d_in_flux_to_sol_full = np.kron(np.diagflat(isoparametric_scale), np.dot(flux_to_sol, d_in_flux))

        # Continuity between cells
        riemann_d = np.eye(self.n_cell * (self.p + 1))
        for i in range(self.n_cell):
            riemann_d[i * (self.p + 1), i * (self.p + 1)] = 0.5
            riemann_d[i * (self.p + 1), i * (self.p + 1) - 1] = 0.5
            riemann_d[i * (self.p + 1) - 1, i * (self.p + 1)] = 0.5
            riemann_d[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0.5

        self._jac_diff = self.d * np.dot(self._d_in_flux_to_sol_full,
                                         np.dot(riemann_d, np.dot(d_in_flux_full,
                                                                  np.dot(riemann_d, self._sol_to_flux_full))))

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

        sol_in_flux_point = np.zeros((self.n_cell, self.p + 1))
        flux_in_flux_point_conv = np.zeros((self.n_cell, self.p + 1))
        flux_in_flux_point_diff = np.zeros((self.n_cell, self.p + 1))
        rhs_in_sol_point = np.zeros((self.n_cell, self.p))

        # Getting solution and the flux for diffusion in flux points
        for i in range(self.n_cell):
            sol_in_flux_point[i] = np.dot(sol_to_flux, y[i * self.p:(i + 1) * self.p])
            flux_in_flux_point_diff[i] = self.d * (np.dot(sol_to_flux, y[i * self.p:(i + 1) * self.p]))

        # Ensuring continuity
        for i in range(self.n_cell):
            yl, yr = sol_in_flux_point[i - 1, -1], sol_in_flux_point[i, 0]
            if yl > yr:  # shock
                if yl + yr > 0:
                    riemann_conv = yl
                elif yl + yr < 0:
                    riemann_conv = yr
                else:
                    riemann_conv = 0
            elif yl < yr:  # rarefaction
                if yl > 0:
                    riemann_conv = yl
                elif yr < 0:
                    riemann_conv = yr
                else:
                    riemann_conv = 0
            else:
                riemann_conv = yl
            sol_in_flux_point[i - 1, -1] = riemann_conv
            sol_in_flux_point[i, 0] = riemann_conv
            riemann_diff = (flux_in_flux_point_diff[i, 0] + flux_in_flux_point_diff[i - 1, -1]) / 2
            flux_in_flux_point_diff[i, 0] = riemann_diff
            flux_in_flux_point_diff[i - 1, -1] = riemann_diff

        # Getting convection flux in flux points
        for i in range(self.n_cell):
            flux_in_flux_point_conv[i] = -sol_in_flux_point[i] * sol_in_flux_point[i] / 2

        # Getting the diffusion flux in flux points
        for i in range(self.n_cell):
            scale = 2 / (self.mesh[i + 1] - self.mesh[i])
            flux_in_flux_point_diff[i] = np.dot(scale * d_in_flux, flux_in_flux_point_diff[i])

        # Ensuring the flux continuity for diffusion
        for i in range(self.n_cell):
            riemann_diff = (flux_in_flux_point_diff[i, 0] + flux_in_flux_point_diff[i - 1, -1]) / 2
            flux_in_flux_point_diff[i, 0] = riemann_diff
            flux_in_flux_point_diff[i - 1, -1] = riemann_diff

        # Getting the rhs in sol points
        for i in range(self.n_cell):
            scale = 2 / (self.mesh[i + 1] - self.mesh[i])
            rhs_in_sol_point[i] = np.dot(scale * d_in_flux_to_sol,
                                         flux_in_flux_point_conv[i] + flux_in_flux_point_diff[i])

        # return rhs_in_sol_point.reshape(y.shape)
        """
        rhs = np.dot(self._riemann_solver(y), np.dot(self._sol_to_flux_full, y))
        rhs = np.dot(self._d_in_flux_to_sol_full, -rhs * rhs / 2) + np.dot(self._jac_diff, y)
        return rhs

    def jac(self, y, t):
        foo = np.dot(self._riemann_solver(y), self._sol_to_flux_full)
        jac_conv = -np.dot(self._d_in_flux_to_sol_full, foo * (np.dot(foo, y)[:, None]))
        return self._jac_diff + jac_conv

    def _riemann_solver(self, y):
        """

        :param array_like y:
        :return: numpy.ndarray - the continuity matrix expressed in the flux points
        """
        y_in_fp = np.dot(self._sol_to_flux_full, y)
        riemann_c = np.eye(self.n_cell * (self.p + 1))
        for i in range(self.n_cell):
            yl, yr = y_in_fp[i * (self.p + 1) - 1], y_in_fp[i * (self.p + 1)]
            if yl > yr:  # shock
                if yl + yr > 0:
                    riemann_c[i * (self.p + 1), i * (self.p + 1)] = 0
                    riemann_c[i * (self.p + 1), i * (self.p + 1) - 1] = 1
                elif yl + yr < 0:
                    riemann_c[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
                    riemann_c[i * (self.p + 1) - 1, i * (self.p + 1)] = 1
                else:
                    riemann_c[i * (self.p + 1), i * (self.p + 1)] = 0
                    riemann_c[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
            elif yl < yr:  # rarefaction
                if yl > 0:
                    riemann_c[i * (self.p + 1), i * (self.p + 1)] = 0
                    riemann_c[i * (self.p + 1), i * (self.p + 1) - 1] = 1
                elif yr < 0:
                    riemann_c[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
                    riemann_c[i * (self.p + 1) - 1, i * (self.p + 1)] = 1
                else:
                    riemann_c[i * (self.p + 1), i * (self.p + 1)] = 0
                    riemann_c[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
        return riemann_c

    def __repr__(self):
        foo = "Spectral difference for Burger's equation " + super(SpectralDifferenceMethodBurgers, self).__repr__()

        foo += "\n\nFlux points [-1, 1] :"
        cell_str = '   '.join(map('{0:0.2f}'.format, self.flux_pts))
        cell = len(cell_str) * ['-']
        for i in self.flux_pts:
            cell[int((len(cell) - 1) * (1 + i) / 2)] = '|'
        foo += "\n{0}\n{1}".format(cell_str, ''.join(cell))
        return foo
