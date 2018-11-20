import numpy as np


class FiniteDifferenceMethod:
    """
    Upwind (as regard of the convection speed) scheme for convection flux
    """

    def __init__(self, mesh, order, conv):
        """

        :param mesh: array_like
        :param order: int
        :param conv: float
        """
        self.mesh = mesh
        self.n_cell = len(mesh) - 1
        self.p = order
        self.n_pts = order * self.n_cell
        self.c = conv

        # Gauss points
        self.cell = np.array([-np.cos(np.pi * (2 * i + 1) / (2 * order)) for i in range(order)])
        # Uniform distribution
        # self.cell = np.array([(2 * i + 1) / order - 1 for i in range(order)])

        self.x = np.zeros(self.n_pts + 1)
        for i in range(self.n_cell):
            scale = self.mesh[i + 1] - self.mesh[i]
            self.x[i * self.p:(i + 1) * self.p] = self.mesh[i] + scale * (self.cell + 1) / 2
        self.x[self.n_pts] = self.mesh[-1] + self.x[0] - self.mesh[0]

    def rhs(self, y, t):
        """
        Return the right hand side for the convection equation

        :param y: array_like
        :param t: float
        :return: numpy.ndarray -
            The RHS, with the same shape as y
        """
        # Can also be written :
        # flux = -self.c * (np.roll(y, -1) - y) / (np.roll(self.x, -1) - self.x)[:-1]
        # if self.c > 0:
        #     flux = np.roll(flux, -1)
        if self.c < 0:
            # Downstream flux
            flux = -self.c * (np.roll(y, -1) - y) / (np.roll(self.x, -1) - self.x)[:-1]
        else:
            # Upstream flux
            flux = -self.c * (y - np.roll(y, 1)) / np.roll((self.x - np.roll(self.x, 1))[1:], 1)
        return flux

    def __repr__(self):
        foo = "Finite difference Method, on [{0}, {1}] (periodic)".format(self.mesh[0], self.mesh[-1])

        dx = (np.roll(self.x, -1) - self.x)[:-1]
        dx_min = min(dx)
        dx_max = max(dx)
        if abs(1 - dx_max / dx_min) < 1E-10:
            foo += "\ndx = {0:0.3E}".format(dx_min)
        else:
            foo += "\ndx_min = {0:0.3E}, dx_max = {1:0.3E}".format(dx_min, dx_max)

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
    p = 5
    c = 1
    mesh1 = np.linspace(0, 10, n)
    method = FiniteDifferenceMethod(mesh1, p, c)
    print(method)
