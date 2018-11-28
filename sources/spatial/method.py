import numpy as np


class SpatialMethod:
    """
    Generic structure for spatial methods on periodic mesh
    """

    def __init__(self, mesh, order, conv):
        """

        :param mesh: array_like
        :param order: int
        :param conv: float
        """
        self.mesh = mesh
        self.n_cell = len(mesh) - 1
        self.p = order + 1
        self.n_pts = self.p * self.n_cell
        self.c = conv

        # Setting the solution points in a [-1, 1] cell as the Gauss points
        self.cell = np.array([-np.cos(np.pi * (2 * i + 1) / (2 * self.p)) for i in range(self.p)])

        # Setting the coordinates of all solutions points
        self.x = np.zeros(self.n_pts)
        for i in range(self.n_cell):
            scale = self.mesh[i + 1] - self.mesh[i]
            self.x[i * self.p:(i + 1) * self.p] = self.mesh[i] + scale * (self.cell + 1) / 2

        # Getting the space steps
        dx = (np.roll(self.x, -1) - self.x)[:-1]
        dx_min = min(dx)
        dx_max = max(dx)
        if abs(1 - dx_max / dx_min) < 1E-10:
            self.dx = (dx_min,)
        else:
            self.dx = (dx_min, dx_max)

    def rhs(self, y, t):
        r"""
        :param y: array_like
        :param t: float
        :return: numpy.ndarray - the right hand side :math:`RHS\left(y, t\right)`
        """

    def jac(self, y, t):
        r"""
        :param y: array_like
        :param t: float
        :return: numpy.ndarray - the jacobian :math:`\frac{\partial RHS}{\partial y}\left(y, t\right)`
        """

    def __repr__(self):
        foo = "Method, on [{0}, {1}] (periodic)".format(self.mesh[0], self.mesh[-1])

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
