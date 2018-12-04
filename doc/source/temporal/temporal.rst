Temporal Methods
================

The temporal method is a function used to solve the equation for a given right hand side:

.. math::
   \left\{\begin{aligned}
    \dot{y}\left(t\right) &= f\left(y, t\right) \\
    y\left(t_0\right) &= y_0
   \end{aligned}\right.


.. py:function:: temporal.temporal_method(y0, t, f, verbose=true, **kwargs)

   :param array_like y0: Initial value, may be multi-dimensional of size d
   :param 1D_array t: Array of time steps, of size n
   :param func f: Function with well shaped input and output
   :param verbose: If True or a string, displays a progress bar
   :type verbose: bool or str, optional
   :param **kwargs: More optional keyword arguments that might be needed
   :return: numpy.ndarray - The solution, of shape (n, d)

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   rk.rst
   bdf.rst
   stability.rst
   test.rst

