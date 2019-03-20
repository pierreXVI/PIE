r"""
This subpackage implements a 1D viscous Burger's equation scheme:
:math:`RHS\left(y, t\right) = -\frac{1}{2}\frac{\partial y^2}{\partial x} + d\frac{\partial^2y}{\partial x^2}`
"""

from .fd_burg import FiniteDifferenceMethodBurgers
from .sd_burg import SpectralDifferenceMethodBurgers
