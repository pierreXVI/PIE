import numpy as np
import scipy.optimize
from rk import rk_4


def Gear_2(y0, t, f):
    """
    Gear method 2nd order
    y' = f(y, t)
    y(t[0]) = y0
    :param y0: initial value, may be multi-dimensional of size d
    :param t: array of time steps, of size n
    :param f: a function with well shaped input and output
    :return: the solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))

    y[0] = y0
    y[1] = y0 + (t[1] - t[0]) * f(y0, t[0])

    for i in range(1, n - 1):
        def g(u):
            return u - (4. / 3. * y[i] - 1. / 3. * y[i - 1] + 2. / 3. * (t[i + 1] - t[i]) * f(u, t[i + 1]))

        y[i + 1] = scipy.optimize.newton(g, y[i])

    return y

import from rk import Euler_explicit

def bdf_4(y0,t0,tf,n,f):

	t=t0
	y=y0
	h=(tf-t0)/float(n)
	Approx_solution=Euler_explicit(f,t0,t0+3*h,y0,3)
	Temps=[t0]
	for i in range(n):
		 t+=h
		 Temps.append(t)
		 def F1(z):
		  return z-(48./25)*Approx_solution[i+3]+(36./25)*Approx_solution[i+2]-(16./25)*Approx_solution[i+1]+(3./25)*Approx_solution[i]=(12./25)*h*f(z,Temps[i+3])
		  Approx_solution[i+4]=scipy.optimize.newton(F1,Approx_solution[i+3])
	plt.plot(Temps,Approx_solution,'b')
	plt.xlabel("Temps")
	plt.ylabel("y(t)")
	plt.show()
return Approx_solution, Temps 	

def bdf_6(y0, t, f):
    """
    Backward Differentiation Formula - 6 method
    y' = f(y, t)
    y(t[0]) = y0

    :param y0: initial value, may be multi-dimensional of size d
    :param t: array of time steps, of size n
    :param f: a function with well shaped input and output
    :return: the solution, of shape (n, d)
    """
    try:
        n, d = len(t), len(y0)
        y = np.zeros((n, d))
    except TypeError:
        n = len(t)
        y = np.zeros((n,))
    y[:6] = rk_4(y0, t[:6], f)
    for i in range(n - 6):
        h = t[i + 1] - t[i]
        l = lambda u: 147. * u - 360. * y[i + 5] + 450. * y[i + 4] - 400. * y[i + 3] + 225. * y[i + 2] - 72. * y[
            i + 1] + 10. * y[i] - 60. * h * f(u, t[i + 1])
        y[i + 6] = scipy.optimize.newton(l, y[i + 5])
    return y


if __name__ == '__main__':
    pass
