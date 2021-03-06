import numpy as np

from ..method import _SpatialMethod


class FiniteDifferenceMethodBurgers(_SpatialMethod):
    r"""
    1D scheme for viscous Burgers' equation with a periodic boundary condition.

    :ivar array_like _j1: The constant jacobian for left convection
    :ivar array_like _j2: The constant jacobian for right convection
    :ivar array_like _jac_diff: The constant jacobian for the diffusion part
    """

    def __init__(self, mesh, p, diff):
        super(FiniteDifferenceMethodBurgers, self).__init__(mesh, p, 0, diff)

        x = np.append(self.x, self.mesh[-1] + self.x[0] - self.mesh[0])
        dx1 = np.roll((x - np.roll(x, 1))[1:], 1)
        dx2 = (np.roll(x, -1) - x)[:-1]
        self.j1 = np.diagflat(1 / dx1) - np.diagflat(1 / dx1[1:], -1)
        self.j1[0, -1] = -1 / dx1[0]
        self.j2 = np.diagflat(1 / dx2[:-1], 1) - np.diagflat(1 / dx2)
        self.j2[-1, 0] = 1 / dx2[-1]
        self._jac_diff = self.d * 2 * (self.j2 - self.j1) / (dx1[:, None] + dx2[:, None])

    def rhs(self, y, t):
        """
        rhs = np.zeros(y.shape)
        for i in range(1, len(y) - 1):
            a = (y[i + 1] - y[i]) / (self.x[i + 1] - self.x[i])
            b = (y[i] - y[i - 1]) / (self.x[i] - self.x[i - 1])
            if y[i] > 0:
                rhs[i] += -y[i] * b
            else:
                rhs[i] += -y[i] * a
            rhs[i] += self.d * 2 * (a - b) / (self.x[i + 1] - self.x[i - 1])

        a = (y[1] - y[0]) / (self.x[1] - self.x[0])
        b = (y[0] - y[-1]) / (self.x[0] - self.x[- 1] + self.mesh[-1])
        if y[0] > 0:
            rhs[0] += -y[0] * b
        else:
            rhs[0] += -y[0] * a
        rhs[0] += self.d * 2 * (a - b) / (self.x[1] - self.x[-1] + self.mesh[-1])

        a = (y[0] - y[-1]) / (self.mesh[-1] + self.x[0] - self.x[-1])
        b = (y[-1] - y[-2]) / (self.x[-1] - self.x[-2])
        if y[-1] > 0:
            rhs[-1] += -y[-1] * b
        else:
            rhs[-1] += -y[-1] * a
        rhs[-1] += self.d * 2 * (a - b) / (self.mesh[-1] + self.x[0] - self.x[-2])
        """
        j = self.j1 * (y > 0)[:, None] + self.j2 * (y < 0)[:, None]
        return np.dot(self._jac_diff, y) - y * np.dot(j, y)

    def jac(self, y, t):
        j = self.j1 * (y > 0)[:, None] + self.j2 * (y < 0)[:, None]
        return self._jac_diff - (y[:, None] * j + np.diagflat(np.dot(j, y)))

    def hess(self, y, t):
        # TODO: check this
        j = self.j1 * (y > 0)[:, None] + self.j2 * (y < 0)[:, None]
        hess = np.zeros((self.n_pts, self.n_pts, self.n_pts))
        for k in range(self.n_pts):
            for l in range(self.n_pts):
                for m in range(self.n_pts):
                    hess[k, l, m] = -j[k, l]
        return hess

    def __repr__(self):
        return "Finite difference for Burgers' equation " + super(FiniteDifferenceMethodBurgers, self).__repr__()
