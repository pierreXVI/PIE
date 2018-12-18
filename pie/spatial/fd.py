import numpy as np
from pie.spatial.method import _SpatialMethod


class FiniteDifferenceMethod(_SpatialMethod):
    r"""
    Upwind (as regard of the convection speed) scheme for convection flux,
    with a periodic boundary condition.

    .. math::
       \frac{\partial y}{\partial x}\ _i \simeq
       \left\{\begin{aligned}
       &\frac{y_i - y_{i-1}}{x_i - x_{i-1}} &\text{if $c \gt 0$}\\ \\
       &\frac{y_{i+1} - y_i}{x_{i+1} - x_i} &\text{otherwise}
       \end{aligned}\right.

    This method gives a linear right hand side so it has a constant jacobian, stored as a private attribute.

    :ivar array_like _jac: The constant jacobian
    """

    def __init__(self, mesh, p, conv):
        super(FiniteDifferenceMethod, self).__init__(mesh, p, conv)

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
        x = np.append(self.x, self.mesh[-1] + self.x[0] - self.mesh[0])
        if self.c < 0:
            # Downstream flux
            rhs = -self.c * (np.roll(y, -1) - y) / (np.roll(x, -1) - x)[:-1]
        else:
            # Upstream flux
            rhs = -self.c * (y - np.roll(y, 1)) / np.roll((x - np.roll(x, 1))[1:], 1)
        # Can also be written :
        # rhs = -self.c * (np.roll(y, -1) - y) / (np.roll(x, -1) - x)[:-1]
        # if self.c > 0:
        #     rhs = np.roll(flux, -1)
        # Or :
        # rhs = np.dot(self._jac, y)
        return rhs

    def jac(self, y, t):
        return self._jac

    def __repr__(self):
        foo = "Finite difference " + super(FiniteDifferenceMethod, self).__repr__()
        return foo
