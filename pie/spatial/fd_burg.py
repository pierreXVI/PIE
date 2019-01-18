import numpy as np
from pie.spatial.method import _SpatialMethod


class FiniteDifferenceMethodBurgers(_SpatialMethod):
    r"""
    Upwind (as regard of the convection speed) scheme for viscous Burgers' equation,
    with a periodic boundary condition.
    """

    def __init__(self, mesh, p, diff):
        super(FiniteDifferenceMethodBurgers, self).__init__(mesh, p, 0, diff)

    def rhs(self, y, t):
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
        return rhs

    def jac(self, y, t):
        # TODO
        pass

    def __repr__(self):
        return "Finite difference for Burgers' equation " + super(FiniteDifferenceMethodBurgers, self).__repr__()
