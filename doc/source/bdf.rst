
Backward Differentiation Formula methods
========================================

`BDF methods <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_ on Wikipedia.


.. py:function:: temporal.bdf_1(y0, t, f, jac=None)
.. py:function:: temporal.bdf_2(y0, t, f, jac=None)
.. py:function:: temporal.bdf_3(y0, t, f, jac=None)
.. py:function:: temporal.bdf_4(y0, t, f, jac=None)
.. py:function:: temporal.bdf_5(y0, t, f, jac=None)
.. py:function:: temporal.bdf_6(y0, t, f, jac=None)

   :param array_like y0: Initial value, may be multi-dimensional of size d
   :param 1D_array t: Array of time steps, of size n
   :param func f: Function with well shaped input and output
   :param jac: If given, the Jacobian of f
   :type jac: func or None, optional
   :param verbose: If True or a string, displays a progress bar
   :type verbose: bool or str, optional
   :return: numpy.ndarray - The solution, of shape (n, d)
