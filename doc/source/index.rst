.. PIE documentation master file, created by
   sphinx-quickstart on Thu Nov  1 18:09:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PIE's documentation!
===============================

The packages documented here are used to solve the 1D-PDE:

.. math::
   \left\{\begin{aligned}
    \frac{\partial y}{\partial t}\left(x, t\right) &= RHS\left(y, x, t\right) \\
    y\left(t_0\right) &= y_0
   \end{aligned}\right.

The package ``temporal`` is used to solve :math:`\dot{y}\left(t\right) = f\left(y, t\right)`
for a given right hand side, and the package ``spatial`` is used to compute a right hand side with x-derivatives.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   temporal/temporal.rst
   spatial/spatial.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
