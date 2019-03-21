Tests on PDE
============

The module ``pie.test.test_pde`` can be run to test some numerical methods
on an 1D advection - diffusion or a Burgers' PDE.
The equation is solved on a periodic window, and with a given :doc:`initial condition <initial_condition_pde>`.

.. automodule:: pie.test.test_pde
   :members: solve, compare, compare_burgers

.. autodata:: pie.test.test_pde.TEMPORAL_METHODS
   :annotation:

.. autodata:: pie.test.test_pde.SPATIAL_METHODS
   :annotation:

.. autodata:: pie.test.test_pde.SPATIAL_METHODS_BURGERS
   :annotation:


