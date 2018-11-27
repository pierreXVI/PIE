import numpy as np
from spatial.method import SpatialMethod

import matplotlib.pyplot as plt
from spatial.fd import FiniteDifferenceMethod


# TODO: Check the flux continuity
# --> should the flux or the RHS be continuous
# --> is the symmetric boundary condition well written
# --> is it working for convection to the left (negative self.c)
# TODO: Write jac(self, y, t)
# TODO: Validate then clean tests

class SpectralDifferenceMethod(SpatialMethod):
    """
    """

    def __init__(self, mesh, order, conv):
        super(SpectralDifferenceMethod, self).__init__(mesh, order, conv)

        # Setting the solution points in a [-1, 1] cell as the Legendre roots
        self.flux_pts = np.append(-1, np.append(np.polynomial.legendre.legroots(order * [0] + [1]), 1))

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

    def rhs(self, y, t):
        sol_in_sol_point = np.zeros(y.shape)
        sol_in_flux_point = np.zeros(y.ndim * [self.n_cell * (self.p + 1)])

        flux_in_sol_point = np.zeros(y.shape)
        flux_in_flux_point = np.zeros(y.ndim * [self.n_cell * (self.p + 1)])

        rhs_in_sol_point = np.zeros(y.shape)
        rhs_in_flux_point = np.zeros(y.ndim * [self.n_cell * (self.p + 1)])

        # Getting solution in sol points
        for i in range(self.n_cell):
            # Getting the isoparametric solution in the cell
            sol = y[i * self.p:(i + 1) * self.p] * 2 / (self.mesh[i + 1] - self.mesh[i])
            sol_in_sol_point[i * self.p:(i + 1) * self.p] = sol

        # Getting solution in flux points
        for i in range(self.n_cell):
            sol_in_flux_point[i * (self.p + 1):(i + 1) * (self.p + 1)] = np.dot(self.sol_to_flux,
                                                                                sol_in_sol_point[i * self.p:
                                                                                                 (i + 1) * self.p])

        # Getting the flux in flux points
        for i in range(self.n_cell):
            a, b = i * (self.p + 1), (i + 1) * (self.p + 1)
            flux_in_flux_point[a:b] = -self.c * sol_in_flux_point[a:b]

        # Setting the flux continuity
        for i in range(self.n_cell):
            if self.c > 0:
                flux_in_flux_point[i * (self.p + 1)] = flux_in_flux_point[i * (self.p + 1) - 1]
            else:
                flux_in_flux_point[(i - 1) * (self.p + 1) - 1] = flux_in_flux_point[(i - 1) * (self.p + 1)]

        # Getting the rhs in flux points
        for i in range(self.n_cell):
            a, b = i * (self.p + 1), (i + 1) * (self.p + 1)
            rhs_in_flux_point[a:b] = np.dot(self.d_in_flux, flux_in_flux_point[a:b])

        # Getting the rhs in sol points
        for i in range(self.n_cell):
            rhs_in_sol_point[i * self.p:(i + 1) * self.p] = np.dot(self.flux_to_sol,
                                                                   rhs_in_flux_point[i * (self.p + 1):
                                                                                     (i + 1) * (self.p + 1)])

        return rhs_in_sol_point

    def rhs_test(self):
        x_sol = np.zeros(self.n_pts)
        x_flux = np.zeros(self.n_pts + self.n_cell)
        for i in range(self.n_cell):
            scale = self.mesh[i + 1] - self.mesh[i]
            x_sol[i * self.p:(i + 1) * self.p] = self.mesh[i] + scale * (self.cell + 1) / 2
            x_flux[i * (self.p + 1):(i + 1) * (self.p + 1)] = self.mesh[i] + scale * (self.flux_pts + 1) / 2

        # y = np.sin(2 * np.pi * x_sol / self.mesh[-1])
        # rhs_expected = -self.c * np.cos(2 * np.pi * x_sol / self.mesh[-1]) * 2 * np.pi / self.mesh[-1]
        y = np.exp(-200. * (x_sol / self.mesh[-1] - 0.3) ** 2)
        rhs_expected = c * 400 * (x_sol / self.mesh[-1] - 0.3) * np.exp(-200. * (x_sol / self.mesh[-1] - 0.3) ** 2) / \
                       self.mesh[-1]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xticks(self.mesh)
        ax1.grid(True)

        sol_in_sol_point = np.zeros(y.shape)
        sol_in_flux_point = np.zeros(y.ndim * [self.n_cell * (self.p + 1)])

        flux_in_sol_point = np.zeros(y.shape)
        flux_in_flux_point = np.zeros(y.ndim * [self.n_cell * (self.p + 1)])

        rhs_in_sol_point = np.zeros(y.shape)
        rhs_in_flux_point = np.zeros(y.ndim * [self.n_cell * (self.p + 1)])

        # Getting solution in sol points
        for i in range(self.n_cell):
            # Getting the isoparametric solution in the cell
            sol = y[i * self.p:(i + 1) * self.p] * 2 / (self.mesh[i + 1] - self.mesh[i])
            sol_in_sol_point[i * self.p:(i + 1) * self.p] = sol
        ax1.plot(x_sol, sol_in_sol_point, '+-', label='Sol in sol')

        # Getting solution in flux points
        for i in range(self.n_cell):
            sol_in_flux_point[i * (self.p + 1):(i + 1) * (self.p + 1)] = np.dot(self.sol_to_flux,
                                                                                sol_in_sol_point[i * self.p:
                                                                                                 (i + 1) * self.p])
        ax1.plot(x_flux, sol_in_flux_point, '+-', label='Sol in flux')

        # Getting the flux in flux points
        for i in range(self.n_cell):
            a, b = i * (self.p + 1), (i + 1) * (self.p + 1)
            flux_in_flux_point[a:b] = -self.c * sol_in_flux_point[a:b]
        ax1.plot(x_flux, flux_in_flux_point, '+-', lw=3, label='Flux in flux')

        # Setting the flux continuity
        for i in range(self.n_cell):
            if self.c > 0:
                flux_in_flux_point[i * (self.p + 1)] = flux_in_flux_point[i * (self.p + 1) - 1]
            else:
                flux_in_flux_point[(i - 1) * (self.p + 1) - 1] = flux_in_flux_point[(i - 1) * (self.p + 1)]
        ax1.plot(x_flux, flux_in_flux_point, '+-', label='Flux_c in flux')

        ax1.plot(x_sol, rhs_expected, 'k--', lw=5, label='RHS expected')

        # Getting the rhs in flux points
        for i in range(self.n_cell):
            a, b = i * (self.p + 1), (i + 1) * (self.p + 1)
            rhs_in_flux_point[a:b] = np.dot(self.d_in_flux, flux_in_flux_point[a:b])
        ax1.plot(x_flux, rhs_in_flux_point, '+-', lw=3, label='RHS in flux')

        # Getting the rhs in sol points
        for i in range(self.n_cell):
            rhs_in_sol_point[i * self.p:(i + 1) * self.p] = np.dot(self.flux_to_sol,
                                                                   rhs_in_flux_point[i * (self.p + 1):
                                                                                     (i + 1) * (self.p + 1)])
        ax1.plot(x_sol, rhs_in_sol_point, '+-', label='RHS in sol')

        ax1.legend()
        plt.show()

    def jac(self, y, t):
        pass

    def __repr__(self):
        foo = "Spectral difference " + super(SpectralDifferenceMethod, self).__repr__()

        foo += "\n\nFlux points [-1, 1] :"
        cell_str = '   '.join(map('{0:0.2f}'.format, self.flux_pts))
        cell = len(cell_str) * ['-']
        for i in self.flux_pts:
            print(i)
            cell[int((len(cell) - 1) * (1 + i) / 2)] = '|'
        foo += "\n{0}\n{1}".format(cell_str, ''.join(cell))
        return foo


def lagrange(x, x_i, i):
    """
    Evaluate in x the i polynomial of the Lagrange base on the points x_i

    Returns :math:`L_i\left(x\right)`
    with :math:`L_i\left(x_i\left[j\right]\right) = \delta_{ij}`

    :param x: float
    :param x_i: array_like
    :param i: int
    :return: float
    """
    foo = np.delete(x_i, i)
    return np.prod((x - foo) / (x_i[i] - foo))


def d_lagrange(x, x_i, i):
    r"""
    Evaluate in x the derivative of the i polynomial of the Lagrange base on the points x_i

    Returns :math:`\frac{\mathrm{d}L_i}{\mathrm{d}x}\left(x\right)`
    with :math:`L_i\left(x_i\left[j\right]\right) = \delta_{ij}`

    :param x: float
    :param x_i: array_like
    :param i: int
    :return: float
    """
    val = 0
    foo = np.delete(x_i, i)
    for k in range(len(foo)):
        val += np.prod(x - np.delete(foo, k))
    return val / np.prod(x_i[i] - foo)


if __name__ == '__main__':
    n = 3
    p = 4
    c = 1
    L = 1
    mesh1 = np.linspace(0, L, n + 1)

    method = SpectralDifferenceMethod(mesh1, p, c)
    # method_fd = FiniteDifferenceMethod(mesh1, p, c)
    print(method)

    # x = method.x[:-1]
    # y0 = np.sin(2 * np.pi * x / L)
    # method.rhs_test()

    # print(method.d_in_flux)

    # x = method.x[:-1]
    # y0 = np.exp(-200. * (x / L - 0.3) ** 2)
    # rhs_0 = c * 400 * (x / L - 0.3) * np.exp(-200. * (x / L - 0.3) ** 2) / L
    # y0 = np.sin(2 * np.pi * x / L)
    # rhs_0 = -c * np.cos(2 * np.pi * x / L) * 2 * np.pi / L
    # y0 = np.sin(4 * np.pi * x / L) * (x > L / 4) * (x < 3 * L / 4)
    # rhs_0 = -c * np.cos(4 * np.pi * x / L) * 4 * np.pi / L * (x > L / 4) * (x < 3 * L / 4)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.grid(True)
    # ax1.plot(x, y0, label='Initial condition')

    # rhs_fd = method_fd.rhs(y0, 0)
    # rhs_sd = method.rhs(y0, 0)

    # ax1.plot(x, rhs_0, 'k+--', lw=5, label='Exact RHS')
    # ax1.plot(x, rhs_fd, '+--', lw=3, label='Finite Difference Method')
    # ax1.plot(x, rhs_sd, '+-', label='Spectral Difference Method')
    # print('Error :')
    # print('FD : {0:0.2E} (max : {1:0.2E})'.format(np.mean(abs(rhs_fd - rhs_0)), np.max(abs(rhs_fd - rhs_0))))
    # print('SD : {0:0.2E} (max : {1:0.2E})'.format(np.mean(abs(rhs_sd - rhs_0)), np.max(abs(rhs_sd - rhs_0))))

    # ax1.legend()
    # plt.show()
