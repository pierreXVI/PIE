Tests on PDE
============

The module ``pie.test.test_pde`` can be run to test some numerical methods
on an 1D advection - diffusion or a Burgers' PDE.

To use this module, add, remove, comment or uncomment any method in ``TEMPORAL_METHODS``,
``SPATIAL_METHODS`` and ``SPATIAL_METHODS_BURGERS``,
then run ``compare`` or ``compare_burgers`` with the desired :doc:`initial condition <initial_condition_pde>`.

.. automodule:: pie.test.test_pde
   :members: solve, compare, compare_burgers

.. autodata:: pie.test.test_pde.TEMPORAL_METHODS
   :annotation:

.. autodata:: pie.test.test_pde.SPATIAL_METHODS
   :annotation:

.. autodata:: pie.test.test_pde.SPATIAL_METHODS_BURGERS
   :annotation:


