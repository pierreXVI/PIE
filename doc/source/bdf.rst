
Backward Differentiation Formula methods
========================================

`BDF methods <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_ on Wikipedia.


.. py:function:: bdf.bdf_1(y0, t, f)
.. py:function:: bdf.bdf_2(y0, t, f)
.. py:function:: bdf.bdf_3(y0, t, f)
.. py:function:: bdf.bdf_4(y0, t, f)
.. py:function:: bdf.bdf_5(y0, t, f)
.. py:function:: bdf.bdf_6(y0, t, f)

    :param y0: array_like -
        Initial value, may be multi-dimensional of size d
    :param t: 1D_array -
        Array of time steps, of size n
    :param f: func -
        Function with well shaped input and output
    :param jac: func or None, optional -
        If given, the Jacobian of f
    :return: numpy.ndarray -
        The solution, of shape (n, d)
