.. PIE documentation master file, created by
   sphinx-quickstart on Thu Nov  1 18:09:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PIE's documentation!
===============================

The different methods are used to solve the ODE:

.. math::
   \left\{\begin{aligned}
    \dot{y}\left(t\right) &= f\left(y, t\right) \\
    y\left(t_0\right) &= y_0
   \end{aligned}\right.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   rk.rst
   bdf.rst
   stability.rst
   test.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
