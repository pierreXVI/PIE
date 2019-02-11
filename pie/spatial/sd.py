import numpy as np

import pie.linalg.lagrange
from .method import _SpatialMethod


class SpectralDifferenceMethod(_SpatialMethod):
    """
    Spatial scheme for convection - diffusion flux,
    with a periodic boundary condition, using the spectral difference method.

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
        sol_to_flux = pie.linalg.lagrange.lagrange_extrapolation_matrix(self.cell, self.flux_pts)
        flux_to_sol = pie.linalg.lagrange.lagrange_extrapolation_matrix(self.flux_pts, self.cell)
        d_in_flux = np.zeros((self.p + 1, self.p + 1))
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                d_in_flux[i, j] = pie.linalg.lagrange.d_lagrange(self.flux_pts[i], self.flux_pts, j)

        # Working with full size matrices
        isoparametric_scale = 2 / (np.roll(self.mesh, -1) - self.mesh)[:-1]
        sol_to_flux_full = np.kron(np.eye(self.n_cell), sol_to_flux)
        d_in_flux_full = np.kron(np.diagflat(isoparametric_scale), d_in_flux)
        d_in_flux_to_sol_full = np.kron(np.diagflat(isoparametric_scale), np.dot(flux_to_sol, d_in_flux))

        # Continuity between cells
        riemann_c = np.eye(self.n_cell * (self.p + 1))
        riemann_d = np.eye(self.n_cell * (self.p + 1))
        for i in range(self.n_cell):
            if self.c > 0:
                riemann_c[i * (self.p + 1), i * (self.p + 1)] = 0
                riemann_c[i * (self.p + 1), i * (self.p + 1) - 1] = 1
            else:
                riemann_c[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0
                riemann_c[i * (self.p + 1) - 1, i * (self.p + 1)] = 1
            riemann_d[i * (self.p + 1), i * (self.p + 1)] = 0.5
            riemann_d[i * (self.p + 1), i * (self.p + 1) - 1] = 0.5
            riemann_d[i * (self.p + 1) - 1, i * (self.p + 1)] = 0.5
            riemann_d[i * (self.p + 1) - 1, i * (self.p + 1) - 1] = 0.5

        self._jac = np.dot(d_in_flux_to_sol_full,
                           -self.c * np.dot(riemann_c, sol_to_flux_full)
                           + self.d * np.dot(riemann_d, np.dot(d_in_flux_full, np.dot(riemann_d, sol_to_flux_full))))

    def rhs(self, y, t):
        """
        # Setting the needed matrices
        sol_to_flux = lagrange_extrapolation_matrix(self.cell, self.flux_pts)
        flux_to_sol = lagrange_extrapolation_matrix(self.flux_pts, self.cell)
        d_in_flux = np.zeros((self.p + 1, self.p + 1))
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                d_in_flux[i, j] = d_lagrange(self.flux_pts[i], self.flux_pts, j)
        d_in_flux_to_sol = np.dot(flux_to_sol, d_in_flux)

        flux_in_flux_point_conv = np.zeros((self.n_cell, self.p + 1))
        flux_in_flux_point_diff = np.zeros((self.n_cell, self.p + 1))
        rhs_in_sol_point = np.zeros((self.n_cell, self.p))

        # Getting the flux in flux points
        for i in range(self.n_cell):
            flux_in_flux_point_conv[i] = -self.c * (np.dot(sol_to_flux, y[i * self.p:(i + 1) * self.p]))
            flux_in_flux_point_diff[i] = self.d * (np.dot(sol_to_flux, y[i * self.p:(i + 1) * self.p]))

        # Ensuring the flux continuity
        for i in range(self.n_cell):
            if self.c > 0:
                flux_in_flux_point_conv[i, 0] = flux_in_flux_point_conv[i - 1, -1]
            else:
                flux_in_flux_point_conv[i - 1, -1] = flux_in_flux_point_conv[i, 0]
            riemann_diff = (flux_in_flux_point_diff[i, 0] + flux_in_flux_point_diff[i - 1, -1]) / 2
            flux_in_flux_point_diff[i, 0] = riemann_diff
            flux_in_flux_point_diff[i - 1, -1] = riemann_diff

        # Getting the flux in flux points for diffusion
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

        return rhs_in_sol_point.reshape(y.shape)
        """
        return np.dot(self._jac, y)

    def jac(self, y, t):
        return self._jac

    def jac2(self, y, t):
        return np.zeros((self.n_pts, self.n_pts, self.n_pts))

    def __repr__(self):
        foo = "Spectral difference " + super(SpectralDifferenceMethod, self).__repr__()

        foo += "\n\nFlux points [-1, 1] :"
        cell_str = '   '.join(map('{0:0.2f}'.format, self.flux_pts))
        cell = len(cell_str) * ['-']
        for i in self.flux_pts:
            cell[int((len(cell) - 1) * (1 + i) / 2)] = '|'
        foo += "\n{0}\n{1}".format(cell_str, ''.join(cell))
        return foo
