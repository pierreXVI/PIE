Example
=======

Here is an example on how to use the ``pie`` package.
The following commands are to be executed in python.

Global features:

.. code-block:: python

   # Imports
   import pie
   import numpy as np

   # Parameters
   n = 50  # number of cells
   p = 4  # number of solution points inside a cell
   x_max = 1  # to solve on [0, x_max] spatially
   t_max, dt = 1, 1E-3  # the maximum iteration time and the time step
   conv, diff = 0.8, 0.005  # the convection and diffusion parameters
   krylov_subspace_dim = 10  # the Krylov subspace dimension (or None not to use the Krylov method)

   # Spatial
   mesh = np.linspace(0, x_max, n + 1)  # to create the mesh

   # Initial condition
   init_cond_function = pie.test.initial_condition.sine(x_max)  # or any numerical function on [0, x_max]

   # Temporal
   t = np.append(np.arange(0, t_max, dt), t_max)  # or any time steps array


Setting the spatial method:

   - Advection diffusion problem:
      .. code-block:: python

         method = pie.spatial.SpectralDifferenceMethod(mesh, p, conv, diff)
   - Burgers' problem:
      .. code-block:: python


        method = pie.spatial.burgers.SpectralDifferenceMethodBurgers(mesh, p, diff)

Solving:

.. code-block:: python

   x = method.x
   y0 = init_cond_function(x)

   # Solving
   y = pie.temporal.taylor_exp_3(  # or any other temporal method
           y0, t, method.rhs,
           jac=method.jac, hess=method.hess,
           krylov_subspace_dim=krylov_subspace_dim,
           verbose='Example'  # or any message, or None
       )

Plotting:

   - Using matplotlib:
      .. code-block:: python

         import matplotlib.pyplot as plt

         fig = plt.figure()
         ax = fig.add_subplot(111)
         # ax.set_xticks(mesh)
         ax.grid(True)
         ax.set_xlabel('x')
         ax.plot(x, y[-1], label='Solution at t = {0}'.format(t_max))
         ax.legend()
         plt.show()

   - Using the package Animation class
      .. code-block:: python

         pie.plot.Animation(t, method.x, [y], x_ticks=mesh, title='Example')
