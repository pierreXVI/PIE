
Runge Kutta methods
===================

`RK methods <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ on Wikipedia.

.. py:function:: rk.rk1_1(y0, t, f)
.. py:function:: rk.rk_2(y0, t, f)
.. py:function:: rk.rk_4(y0, t, f)

   :param array_like y0: Initial value, may be multi-dimensional of size d
   :param 1D_array t: Array of time steps, of size n
   :param func f: Function with well shaped input and output
   :param verbose: If True or a string, displays a progress bar
   :type verbose: bool or str, optional
   :return: numpy.ndarray - The solution, of shape (n, d)


.. autofunction:: temporal.rk_butcher

.. autodata:: temporal.rk.A_RK4
   :annotation:
.. autodata:: temporal.rk.B_RK4
   :annotation:
