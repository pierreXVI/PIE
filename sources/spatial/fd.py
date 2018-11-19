import numpy as np


# import matplotlib.pyplot as plt


class FiniteDifferenceMethod:
    def __init__(self, mesh, order, conv, diff=0):
        self.mesh = mesh
        self.n_cell = len(mesh) - 1
        self.p = order
        self.n_pts = order * self.n_cell
        self.c = conv
        self.d = diff

        # If solution points are Gauss points, method is unstable
        # Therefore we take a uniform distribution
        self.cell = np.array([-np.cos(np.pi * (2 * i + 1) / (2 * order)) for i in range(order)])
        # self.cell = np.array([(2 * i + 1) / order - 1 for i in range(order)])

        self.x = np.zeros(self.n_pts + 1)
        for i in range(self.n_pts):
            scale = (self.mesh[1 + i // self.p] - self.mesh[i // self.p])
            self.x[i] = self.mesh[i // self.p] + scale * (self.cell[i % self.p] + 1) / 2
        self.x[self.n_pts] = self.mesh[-1] + self.x[0] - self.mesh[0]
        self.dx = (np.roll(self.x, -1) - self.x)[:-1]
        print(self)

    def rhs(self, y, t):
        #
        # flux = (np.roll(y, -1) - y) / self.dx
        flux = (y - np.roll(y, 1)) / self.dx
        return -self.c * flux

    def __repr__(self):
        foo = "Finite difference Method, on [{0}, {1}] (periodic)".format(self.mesh[0], self.mesh[-1])
        dx_min = min(self.dx)
        dx_max = max(self.dx)
        if abs(1 - dx_max / dx_min) < 1E-10:
            foo += "\ndx = {0}".format(dx_min)
        else:
            foo += "\ndx_min = {0}, dx_max = {1}".format(dx_min, dx_max)
        foo += "\nn_cell = {0}, n_pts = {1}, p = {2} ".format(self.n_cell, self.n_pts, self.p)
        cell_str = '   '.join(map('{0:0.2f}'.format, self.cell))
        cell = len(cell_str) * ['-']
        for i in self.cell:
            cell[int(len(cell_str) * (1 + i) / 2)] = '|'
        foo += "\nCell [-1, 1] :\n[{0}]\n[{1}]".format(cell_str, ''.join(cell))
        return foo


if __name__ == '__main__':
    n = 100
    p = 10
    c = 1
    mesh1 = np.linspace(0, 2 * np.pi, n)
    method = FiniteDifferenceMethod(mesh1, p, c)
    # y = np.sin(method.x[:-1])
