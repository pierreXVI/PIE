import numpy as np


class _SpatialMethod(object):
    r"""
    Generic structure for spatial methods on periodic mesh

    :param array_like mesh:
    :param int p:
    :param float conv:
    :param float diff:

    :ivar array_like mesh: The mesh, an array of size `n_cell + 1`
    :ivar int n_cell: The number of cells
    :ivar int p: The number of points inside a cell
    :ivar int n_pts: The total number of solution points, is equal to `n_cell` \* `p`
    :ivar float c: The convection parameter
    :ivar float d: The diffusion parameter
    :ivar numpy.ndarray cell: The repartition of the solution points inside a [-1, 1] cell
    :ivar numpy.ndarray x: The position of all the solution points
    :ivar tuple dx: The smallest and the biggest space steps. If they are the same, this tuple contains only one element
    """

    def __init__(self, mesh, p, conv, diff):
        self.mesh = mesh
        self.n_cell = len(mesh) - 1
        self.p = p
        self.n_pts = self.p * self.n_cell
        self.c = conv
        self.d = diff

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
        :param array_like y:
        :param float t:
        :return: numpy.ndarray - the right hand side :math:`RHS\left(y, t\right)`
        """

    def jac(self, y, t):
        r"""
        :param array_like y:
        :param float t:
        :return: numpy.ndarray - the jacobian :math:`\frac{\partial RHS}{\partial y}\left(y, t\right)`
        """

    def hess(self, y, t):
        r"""
        :param array_like y:
        :param float t:
        :return: numpy.ndarray - the hessian :math:`\frac{\partial^2RHS}{\partial y^2}\left(y,t\right)`
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
