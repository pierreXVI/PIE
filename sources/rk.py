import numpy as np


def rk_1(y0, t, f):
    """
    RK1 or Explicit Euler method
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
    for i in range(n - 1):
        y[i + 1] = y[i] + (t[i + 1] - t[i]) * f(y[i], t[i])
    return y


def rk_2(y0, t, f):
    """
    RK2 or midpoint method
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
    for i in range(n - 1):
        h = (t[i + 1] - t[i])
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        y[i + 1] = y[i] + h * k2
    return y


def rk_4(y0, t, f):
    """
    RK4 method
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
    for i in range(n - 1):
        h = (t[i + 1] - t[i])
        k1 = f(y[i], t[i])
        k2 = f(y[i] + h * k1 / 2, t[i] + h / 2)
        k3 = f(y[i] + h * k2 / 2, t[i] + h / 2)
        k4 = f(y[i] + h * k3, t[i] + h)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y



import matplotlib.pyplot as plt
import math
import scipy as sc
import numpy as np 

global t 
global Temps1
global Temps2
Temps1=[]
Temps2=[]

def F(y,t):
	return 2*y+0*t

def Sol_Exacte(y0,t0,tf,n):
	t=t0
	y=y0
	global Temps2
	Temps2=[t0]
	Sol_Exacte=[y0]
	for i in range(n):
		t+=0.1
		y=np.exp(2*t)
		Temps2.append(t)
		Sol_Exacte.append(y)
	#plt.plot(Temps,Sol_Exacte,'r')
	#plt.show()
	return Sol_Exacte

def Euler_explicit(f,t0,tf,y0,n):
	t=t0
	y=y0
	h=(tf-t0)/float(n)
	global Temps1
	Temps1=[t0]
	Approx_solution = [y0]

	for i in range(n):
		y += h*f(y,t)
		t += h 
		Approx_solution.append(y)
		Temps1.append(t)

	#plt.plot(Temps,Approx_solution,'b')
	plt.xlabel("Temps")
	plt.ylabel("y(t)")
	#plt.show()
	return Approx_solution, Temps1

Euler_explicit(F,0,5,1,200)
Sol_Exacte(1,0,5,50)

plt.plot(Temps1,Euler_explicit(F,0,5,1,200),'b',label="Sol_Aprox")
plt.plot(Temps2,Sol_Exacte(1,0,5,50),'r',label="Sol_Exacte")
plt.legend(loc='upper left')
plt.show()
