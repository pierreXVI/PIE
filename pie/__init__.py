r"""
The package documented here is used to solve the 1D-PDE:

.. math::
   \left\{\begin{aligned}
    \frac{\partial y}{\partial t}\left(x, t\right) &= RHS\left(y, t\right) \\
    y\left(t_0\right) &= y_0
   \end{aligned}\right.

The subpackage ``temporal`` is used to solve :math:`\dot{y}\left(t\right) = f\left(y, t\right)`
for a given right hand side, and the subpackage ``spatial`` is used to compute a right hand side with x-derivatives.
"""

from . import temporal
from . import spatial
from . import test
