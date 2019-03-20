Exponential methods
===================

`Exponential integrators <https://en.wikipedia.org/wiki/Exponential_integrator>`_ on Wikipedia.

This module implements classic Taylor's methods and Rosenbrock methods.


Order 1 methods:

.. py:function:: temporal.taylor_exp_1(y0, t, f, jac, verbose=True, krylov_subspace_dim=None, **_)
.. py:function:: temporal.rosen_exp_1(y0, t, f, jac, verbose=True, krylov_subspace_dim=None, **_)

   :param func jac: The Jacobian of f, must return an array
   :param krylov_subspace_dim: If given, uses the :doc:`Krylov subspace approximation method<../linalg/krylov>`
   :type krylov_subspace_dim: None or int, optional


Order 2 methods:

.. py:function:: temporal.taylor_exp_2(y0, t, f, jac, df_dt=None, verbose=True, krylov_subspace_dim=None, **_)
.. py:function:: temporal.rosen_exp_2(y0, t, f, jac, df_dt=None, verbose=True, krylov_subspace_dim=None, **_)

   :param func jac: The Jacobian of f, must return an array
   :param df_dt: The f partial derivative with respect to time
   :type df_dt: func or None, optional
   :param krylov_subspace_dim: If given, uses the :doc:`Krylov subspace approximation method<../linalg/krylov>`
   :type krylov_subspace_dim: None or int, optional


Order 3 methods:

.. py:function:: temporal.taylor_exp_3(y0, t, f, jac, jac2, df_dt=None, d2f_dt2=None, d2f_dtdu=None, verbose=True, krylov_subspace_dim=None,**_)
.. py:function:: temporal.rosen_exp_3(y0, t, f, jac, jac2, df_dt=None, d2f_dt2=None, d2f_dtdu=None, verbose=True, krylov_subspace_dim=None,**_)

   :param func jac: The Jacobian of f, must return an array
   :param func jac2: The Hessian of f, must return an array
   :param df_dt: The f partial derivative with respect to time
   :type df_dt: func or None, optional
   :param d2f_dt2: The f second-order partial derivative with respect to time
   :type d2f_dt2: func or None, optional
   :param d2f_dtdu: The f crossed partial derivative, must return an array
   :type d2f_dtdu: func or None, optional
   :param krylov_subspace_dim: If given, uses the :doc:`Krylov subspace approximation method<../linalg/krylov>`
   :type krylov_subspace_dim: None or int, optional
