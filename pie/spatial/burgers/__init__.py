r"""
This subpackage implements spatial method for viscous Burger's equation :
:math:`RHS\left(y, t\right) = -y\frac{\partial y}{\partial x} + d\frac{\partial^2y}{\partial x^2}`
"""

from .fd_burg import FiniteDifferenceMethodBurgers
from .sd_burg import SpectralDifferenceMethodBurgers

# TODO: check if the shock moves at the right speed
