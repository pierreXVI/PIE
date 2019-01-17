r"""
A spatial method is a class used to solve the CFD problem :
:math:`\frac{\partial y}{\partial t} = RHS\left(y, t\right)`

The spatial method is used to compute the right hand side of the equation.
For now, we look at a convection - diffusion equation, and therefore
:math:`RHS\left(y, t\right) = -c\frac{\partial y}{\partial x} + d\frac{\partial^2y}{\partial x^2}`
"""

from .fd import FiniteDifferenceMethod
from .sd import SpectralDifferenceMethod

from .fd_burg import FiniteDifferenceMethodBurgers
from .sd_burg import SpectralDifferenceMethodBurgers
