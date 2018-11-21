Spatial Methods
===============

A spatial method is a class used to solve the CFD problem :
:math:`\frac{\partial y}{\partial t} = RHS\left(y, t\right)`

The spatial method is used to compute the right hand side of the equation. For now, we look at a convection equation, and therefore
:math:`RHS\left(y, t\right) = -c\frac{\partial y}{\partial x}`

.. py:class:: SpatialMethod(mesh, order, conv)

   .. py:method:: rhs(y, t)

      :param y: array_like
      :param t: float
      :return: numpy.ndarray - the right hand side :math:`RHS\left(y, t\right)`

   .. py:method:: jac(y, t)

      :param y: array_like
      :param t: float
      :return: numpy.ndarray - the jacobian :math:`\frac{\partial RHS}{\partial y}\left(y, t\right)`

   .. py:attribute:: mesh

      The mesh, an array of size `n_cell + 1`

   .. py:attribute:: n_cell

   .. py:attribute:: order

      The number of solution points inside a cell

   .. py:attribute:: n_pts

      The total number of solution points, is equal to `n_cell` \* `p`

   .. py:attribute:: c

      The convection parameter

   .. py:attribute:: cell

      The repartition of the solution points inside a [-1, 1] cell

   .. py:attribute:: x

      The position of all the solution points

   .. py:attribute:: dx

      The smallest and the biggest space steps. If they are the same, this tuple contains only one element

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   fd.rst
