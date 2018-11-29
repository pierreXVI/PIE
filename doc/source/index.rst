.. PIE documentation master file, created by
   sphinx-quickstart on Thu Nov  1 18:09:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PIE's documentation!
===============================

The packages documented here are used to solve the PDE:

.. math::
   \left\{\begin{aligned}
    \dot{y}\left(t\right) &= RHS\left(y, t\right) \\
    y\left(t_0\right) &= y_0
   \end{aligned}\right.

The package ``temporal`` is used to solve :math:`\dot{y}\left(t\right) = f\left(y, t\right)`
for an explicit right hand side, and the package ``spatial`` is used to compute a right hand side with x-derivatives.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   temporal.rst
   spatial.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
