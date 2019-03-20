r"""
A spatial method is a class used to solve the CFD problem :
:math:`\frac{\partial y}{\partial t} = RHS\left(y, t\right)`

A spatial method is used to compute the right hand side of the equation.
This package implement a 1D convection - diffusion scheme, and therefore:
:math:`RHS\left(y, t\right) = -c\frac{\partial y}{\partial x} + d\frac{\partial^2y}{\partial x^2}`

Inside a cell of the mesh, the points are placed at the
`Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`_.
"""

from .fd import FiniteDifferenceMethod
from .sd import SpectralDifferenceMethod
