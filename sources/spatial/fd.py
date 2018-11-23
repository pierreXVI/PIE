import numpy as np


class FiniteDifferenceMethod:
    """
    Upwind (as regard of the convection speed) scheme for convection flux,
    with a periodic boundary condition.
    """

    def __init__(self, mesh, order, conv, distribution='Gauss'):
        """
        Desc
        :param mesh: array_like
        :param order: int
        :param conv: float
        """
        self.mesh = mesh
        self.n_cell = len(mesh) - 1
        self.p = order + 1
        self.n_pts = self.p * self.n_cell
        self.c = conv

        # Setting the solution points in a [-1, 1] cell
        if distribution == 'Gauss':
            # Gauss points
            self.cell = np.array([-np.cos(np.pi * (2 * i + 1) / (2 * self.p)) for i in range(self.p)])
        else:
            # Uniform distribution
            self.cell = np.array([(2 * i + 1) / self.p - 1 for i in range(self.p)])

        # Setting the coordinates of all solutions points, with a phantom point at the end
        self.x = np.zeros(self.n_pts + 1)
        for i in range(self.n_cell):
            scale = self.mesh[i + 1] - self.mesh[i]
            self.x[i * self.p:(i + 1) * self.p] = self.mesh[i] + scale * (self.cell + 1) / 2
        self.x[self.n_pts] = self.mesh[-1] + self.x[0] - self.mesh[0]

        # Getting the space steps
        dx = (np.roll(self.x, -1) - self.x)[:-1]
        dx_min = min(dx)
        dx_max = max(dx)
        if abs(1 - dx_max / dx_min) < 1E-10:
            self.dx = (dx_min,)
        else:
            self.dx = (dx_min, dx_max)

        # Setting the RHS jacobian, constant here
        if self.c < 0:
            dx = 1 / (np.roll(self.x, -1) - self.x)[:-1]
            j = np.diagflat(dx[:-1], 1) - np.diagflat(dx)
            j[-1, 0] = dx[-1]
            self._jac = -self.c * j
        else:
            dx = 1 / np.roll((self.x - np.roll(self.x, 1))[1:], 1)
            j = np.diagflat(dx) - np.diagflat(dx[1:], -1)
            j[0, -1] = -dx[0]
            self._jac = -self.c * j

    def rhs(self, y, t):
        r"""
        :param y: array_like
        :param t: float
        :return: numpy.ndarray - the right hand side :math:`RHS\left(y, t\right)`
        """
        # Can also be written :
        # flux = -self.c * (np.roll(y, -1) - y) / (np.roll(self.x, -1) - self.x)[:-1]
        # if self.c > 0:
        #     flux = np.roll(flux, -1)
        # Or :
        # flux = np.dot(self._jac, y)
        if self.c < 0:
            # Downstream flux
            flux = -self.c * (np.roll(y, -1) - y) / (np.roll(self.x, -1) - self.x)[:-1]
        else:
            # Upstream flux
            flux = -self.c * (y - np.roll(y, 1)) / np.roll((self.x - np.roll(self.x, 1))[1:], 1)
        return flux

    def jac(self, y, t):
        r"""
        :param y: array_like
        :param t: float
        :return: numpy.ndarray - the jacobian :math:`\frac{\partial RHS}{\partial y}\left(y, t\right)`
        """

        return self._jac

    def __repr__(self):
        foo = "Finite difference Method, on [{0}, {1}] (periodic)".format(self.mesh[0], self.mesh[-1])

        if len(self.dx) == 2:
            foo += "\ndx_min = {0:0.3E}, dx_max = {1:0.3E}".format(*self.dx)
        else:
            foo += "\ndx = {0:0.3E}".format(*self.dx)

        foo += "\nn_cell = {0}, n_pts = {1}, p = {2} ".format(self.n_cell, self.n_pts, self.p)

        foo += "\nCell [-1, 1] :"
        cell_str = '   '.join(map('{0:0.2f}'.format, self.cell))
        cell = len(cell_str) * ['-']
        for i in self.cell:
            cell[int(len(cell_str) * (1 + i) / 2)] = '|'
        foo += "\n[{0}]\n[{1}]".format(cell_str, ''.join(cell))
        return foo


if __name__ == '__main__':
    n = 100
    order = 3
    c = 1
    mesh1 = np.linspace(0, 10, n)
    method = FiniteDifferenceMethod(mesh1, order, c)
    print(method)
