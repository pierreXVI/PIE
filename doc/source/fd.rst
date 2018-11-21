
Finite Difference Method
========================

.. autoclass:: spatial.fd.FiniteDifferenceMethod

   .. math::
      \frac{\partial y}{\partial x}\ _i \simeq
      \left\{\begin{aligned}
       &\frac{y_i - y_{i-1}}{x_i - x_{i-1}} &\text{if $c \gt 0$}\\ \\
       &\frac{y_{i+1} - y_i}{x_{i+1} - x_i} &\text{otherwise}
      \end{aligned}\right.

   As this method is linear in y with a constant jacobian, it has the private attribute 

   .. py:attribute:: _jac

      The constant jacobian
