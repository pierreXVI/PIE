import math

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy import misc
from pie import utils
from pie import exp_rk

def sol_exa(t,u0,t0):
   # res = np.zeros(len(t))
   # for i in range(len(t)):
        #res[i] = u0/(1+u0*(t[i]-t0))
        #
    return u0*np.exp((t-t0)) #res

def f(u, t):
    #return -u
    return u
    #return u*u    

def jacf(u, t):
    return utils.jacobian(f,u0,t0, eps=1E-5)

def dfdt(u, t):
    return 0

def dijacdti(i, u, t):
    return 0


def expanded_matrix(u0, t0, f, jacf, dfdt, p=2):
    # Euler
    # Semilinear problem
    try:
        d = len(u0)
        fi = np.zeros((d, p))
        ji = np.zeros((d, d, p))
        wi = np.zeros((d, p))
        W = np.zeros((d, p))
        O = np.zeros((p, d))
        a_expanded = np.zeros((p+d, p+d))
    except TypeError:
        fi = np.zeros((1, p))
        ji = np.zeros((1, 1, p))
        wi = np.zeros((1, p))
        W = np.zeros((1, p))
        O = np.zeros((p, 1))
        a_expanded = np.zeros((p + 1, p + 1))

    J = np.zeros((p, p))

    a0 = jacf(u0, t0)

    fi[:, 0] = f(u0, t0) - np.dot(a0,u0)
    # ji[:, :, 0] c deja 0
    wi[:, 0] = fi[:, 0]
    wi[:, 1] = ji[:, :, 0]*fi[:, 0] + dfdt(u0, t0)     # partielle(dg/du)*du/dt + partielle(dg/dt)

    for i in range(p - 1):
        W[:, i] = wi[:, i]
        J[i, i + 1] = 1
    a_expanded[0, 0] = a0
    a_expanded[0, 1] = W[0, 0]
    a_expanded[0, 2] = W[0, 1]
    a_expanded[1, 1] = J[0, 0]
    a_expanded[1, 2] = J[0, 1]
    a_expanded[2, 1] = J[1, 0]
    a_expanded[2, 2] = J[1, 1]

    return a_expanded

def expanded_vector(u, p=2):
    # Euler
    # Semilinear problem
    try:
        d = len(u)
        v = np.zeros((p + d, 1))
        v[0:d-1, :] = u
        v[p+d-1, :] = u
    except TypeError:
        v = np.zeros((p + 1, 1))
        v[0, :] = u
        v[p, :] = u
    return v

def expanded_method(u0, t0, f, jacf, dfdt, h, p=2):
    try:
        d = len(u0)
        I = np.zeros((d, p + d))
        for i in range(d - 1):
            I[i, i] = 1
        # u = np.dot(I, np.exp(h*expanded_matrix(u0, t0, f, jacf, dfdt)), expanded_vector(u0))
    except TypeError:
        I = np.zeros((1, p + 1))
        I[0, 0] = 1
    return np.dot(I,np.dot(np.exp(h*expanded_matrix(u0, t0, f, jacf, dfdt)), expanded_vector(u0)))


if __name__ == '__main__':
    pass
    # test_phi_1()
    #test_jacobian()
    #print(expanded_matrix(1, 0, f, jacf, dfdt))
    #print(expanded_vector(1))
    #print(expanded_method(1, 0, f, jacf, dfdt, 0.01))
    #print(1/(1-0.01))
    t0 = 0;
    u0 = 10;
    t = np.arange(0,10,0.1)

    u_exa = sol_exa(t,u0,t0)

    u_expd = np.zeros(len(t));
    for i in range(len(t)):
        u_expd[i]= expanded_method(u0, t0, f, jacf, dfdt, t[i]) # Méthode matrice augmenter 

    u_approx = exp_rk.exp_euler(u0, t, f)   # méthode Pierre 
    
    plt.figure()
    plt.plot(t,u_approx, label="u_approx", linestyle=':', color='b')
    plt.plot(t,u_exa, label="u_exa", color='r', linestyle="-.")
    plt.plot(t,u_expd, label="u_expd", linestyle='--',color='g')
    plt.legend()
    plt.show()
