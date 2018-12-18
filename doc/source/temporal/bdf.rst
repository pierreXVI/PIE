
Backward Differentiation Formula methods
========================================

`BDF methods <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_ on Wikipedia.


.. py:function:: temporal.bdf_1(y0, t, f, jac=None, verbose=true)
.. py:function:: temporal.bdf_2(y0, t, f, jac=None, verbose=true)
.. py:function:: temporal.bdf_3(y0, t, f, jac=None, verbose=true)
.. py:function:: temporal.bdf_4(y0, t, f, jac=None, verbose=true)
.. py:function:: temporal.bdf_5(y0, t, f, jac=None, verbose=true)
.. py:function:: temporal.bdf_6(y0, t, f, jac=None, verbose=true)

   :param jac: If given, the Jacobian of f
   :type jac: func or None, optional