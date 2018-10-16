import numpy as np

def rk_1(y0, t, f):
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
    y[1] = y0+ (t[1]-t[1])*f(y0,t[0])
    
    for i in range(1,n - 1):
        y[i + 1] = 4./3.*y[i]-1./3.*y[i-1] + 2./3.(t[i + 1] - t[i]) * f(y[i], t[i])
return y

if __name__ == '__main__':
    pass
