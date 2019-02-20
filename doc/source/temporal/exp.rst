Exponential methods
===================


.. py:function:: temporal.taylor_exp_1(y0, t, f, jac, verbose=True, krylov_subspace_dim=None, **_)
.. py:function:: temporal.rosen_exp_1(y0, t, f, jac, verbose=True, krylov_subspace_dim=None, **_)
.. py:function:: temporal.taylor_exp_2(y0, t, f, jac, df_dt=None, verbose=True, krylov_subspace_dim=None, **_)
.. py:function:: temporal.rosen_exp_2(y0, t, f, jac, df_dt=None, verbose=True, krylov_subspace_dim=None, **_)
.. py:function:: temporal.taylor_exp_3(y0, t, f, jac, jac2, df_dt=None, d2f_dt2=None, d2f_dtdu=None, verbose=True, krylov_subspace_dim=None,**_)
.. py:function:: temporal.rosen_exp_3(y0, t, f, jac, jac2, df_dt=None, d2f_dt2=None, d2f_dtdu=None, verbose=True, krylov_subspace_dim=None,**_)

   :param func jac: The Jacobian of f, must return an array
   :param func jac2: The second-order Jacobian of f, must return an array
   :param df_dt: The f partial derivative with respect to time
   :type df_dt: func or None, optional
   :param d2f_dt2: The f second-order partial derivative with respect to time
   :type d2f_dt2: func or None, optional
   :param d2f_dtdu: The f crossed partial derivative, must return an array
   :type d2f_dtdu: func or None, optional
   :param krylov_subspace_dim:
   :type krylov_subspace_dim: None or int, optional
