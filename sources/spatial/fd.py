import numpy as np
import matplotlib.pyplot as plt


class FiniteDifferenceMethod:
    def __init__(self, mesh, order, conv, diff=0):
        self.mesh = mesh
        self.n_cell = len(mesh)
        self.p = order
        self.n_pts = order * (len(mesh) - 1)
        self.c = conv
        self.d = diff

        self.cell = np.array([-np.cos(np.pi * (2 * i + 1) / (2 * order)) for i in range(order)])
        # self.cell = np.array([(2 * i + 1) / order - 1 for i in range(order)])

        self.x = np.zeros(self.n_pts + 1)
        for i in range(self.n_pts):
            scale = (self.mesh[1 + i // self.p] - self.mesh[i // self.p])
            self.x[i] = self.mesh[i // self.p] + scale * (self.cell[i % self.p] + 1) / 2
        self.x[self.n_pts] = self.mesh[-1] + self.x[0] - self.mesh[0]
        self.dx = (np.roll(self.x, -1) - self.x)[:-1]

    def rhs(self, y, t):
        flux = (np.roll(y, -1) - y) / self.dx
        return -self.c * flux


if __name__ == '__main__':
    n = 100
    p = 2
    c = 1
    mesh = np.linspace(0, 2 * np.pi, n)
    method = FiniteDifferenceMethod(mesh, p, c)
    # y = np.sin(method.x[:-1])
