
Backward Differentiation Formula methods
========================================

`BDF methods <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_ on Wikipedia.


.. py:function:: bdf.bdf_1(y0, t, f)
.. py:function:: bdf.bdf_2(y0, t, f)
.. py:function:: bdf.bdf_3(y0, t, f)
.. py:function:: bdf.bdf_4(y0, t, f)
.. py:function:: bdf.bdf_5(y0, t, f)
.. py:function:: bdf.bdf_6(y0, t, f)

    :param y0: initial value, may be multi-dimensional of size d
    :param t: array of time steps, of size n
    :param f: a function with well shaped input and output
    :return: the solution, of shape (n, d)


