
Runge Kutta methods
===================

`RK methods <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ on Wikipedia.

.. py:function:: rk.rk1_1(y0, t, f)
.. py:function:: rk.rk_2(y0, t, f)
.. py:function:: rk.rk_4(y0, t, f)

    :param y0: initial value, may be multi-dimensional of size d
    :param t: array of time steps, of size n
    :param f: a function with well shaped input and output
    :return: the solution, of shape (n, d)


.. autofunction:: rk.rk_butcher

.. autodata:: rk.A_RK4
   :annotation:
.. autodata:: rk.B_RK4
   :annotation:
