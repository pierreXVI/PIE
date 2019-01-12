import numpy as np
from pie.spatial.method import _SpatialMethod


class FiniteDifferenceMethod(_SpatialMethod):
    r"""
    Upwind (as regard of the convection speed) scheme for convection - diffusion flux,
    with a periodic boundary condition.

    .. math::
       \frac{\partial y}{\partial x}\ _i \simeq
       \left\{\begin{aligned}
       &\frac{y_i - y_{i-1}}{x_i - x_{i-1}} &\text{if $c \gt 0$}\\ \\
       &\frac{y_{i+1} - y_i}{x_{i+1} - x_i} &\text{otherwise}
       \end{aligned}\right.

    .. math::
       \frac{\partial^2 y}{\partial x^2}\ _i \simeq
       \frac{2}{x_{i+1} - x_{i-1}}\left(\frac{y_{i+1} - y_i}{x_{i+1} - x_i} - \frac{y_i - y_{i-1}}{x_i - x_{i-1}}\right)

    This method gives a linear right hand side so it has a constant jacobian, stored as a private attribute.

    :ivar array_like _jac: The constant jacobian
    """

    def __init__(self, mesh, p, conv, diff):
        super(FiniteDifferenceMethod, self).__init__(mesh, p, conv, diff)

        # Setting the RHS jacobian, constant here
        x = np.append(self.x, self.mesh[-1] + self.x[0] - self.mesh[0])
        dx1 = np.roll((x - np.roll(x, 1))[1:], 1)
        dx2 = (np.roll(x, -1) - x)[:-1]
        j1 = np.diagflat(1 / dx1) - np.diagflat(1 / dx1[1:], -1)
        j1[0, -1] = -1 / dx1[0]
        j2 = np.diagflat(1 / dx2[:-1], 1) - np.diagflat(1 / dx2)
        j2[-1, 0] = 1 / dx2[-1]
        if self.c < 0:
            self._jac = -self.c * j2 + self.d * 2 * (j2 - j1) / (dx1 + dx2)
        else:
            self._jac = -self.c * j1 + self.d * 2 * (j2 - j1) / (dx1 + dx2)

    def rhs(self, y, t):
        return np.dot(self._jac, y)

    def jac(self, y, t):
        return self._jac

    def __repr__(self):
        return "Finite difference " + super(FiniteDifferenceMethod, self).__repr__()
