import numpy as np
from pie.spatial.method import _SpatialMethod
from pie.spatial import sd


class SpectralDifferenceMethodBurgers(_SpatialMethod):
    """
    Spatial scheme for viscous Burgers' equation,
    with a periodic boundary condition, using the spectral difference method.

    This method uses p + 1 flux points in the [-1, 1] cell, computed as the Legendre polynomial roots.

    :ivar array_like flux_pts: The repartition of the flux points inside a [-1, 1] cell
    """

    def __init__(self, mesh, p, diff):
        super(SpectralDifferenceMethodBurgers, self).__init__(mesh, p, 0, diff)

        # Setting the solution points in a [-1, 1] cell as the Legendre roots
        self.flux_pts = np.append(-1, np.append(np.polynomial.legendre.legroots((self.p - 1) * [0] + [1]), 1))

    def rhs(self, y, t):
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
        return rhs_in_sol_point.reshape(y.shape)

    def jac(self, y, t):
        # TODO
        pass

    def __repr__(self):
        foo = "Spectral difference for Burger's equation " + super(SpectralDifferenceMethodBurgers, self).__repr__()

        foo += "\n\nFlux points [-1, 1] :"
        cell_str = '   '.join(map('{0:0.2f}'.format, self.flux_pts))
        cell = len(cell_str) * ['-']
        for i in self.flux_pts:
            cell[int((len(cell) - 1) * (1 + i) / 2)] = '|'
        foo += "\n{0}\n{1}".format(cell_str, ''.join(cell))
        return foo
