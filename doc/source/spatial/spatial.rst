Spatial Methods
===============

A spatial method is a class used to solve the CFD problem :
:math:`\frac{\partial y}{\partial t} = RHS\left(y, t\right)`

The spatial method is used to compute the right hand side of the equation. For now, we look at a convection equation, and therefore
:math:`RHS\left(y, t\right) = -c\frac{\partial y}{\partial x}`

.. autoclass:: spatial.method._SpatialMethod
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   fd.rst
   sd.rst
