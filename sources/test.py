import matplotlib.pyplot as plt
import numpy as np
import rk
import bdf


def compare_methods():
    """
    Compare RK1, RK2 and RK4 methods on:
    y' = y * sin(t)**2
    y(0) = 1
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)

    x = np.linspace(0, 10, 100)
    y0 = 1

    def f(y, t):
        return y * (np.sin(t) ** 2)

    y_exact = np.exp(x / 2 - np.sin(2 * x) / 4)
    y1 = rk.rk_1(y0, x, f)
    y2 = rk.rk_2(y0, x, f)
    y4 = rk.rk_4(y0, x, f)

    ax1.plot(x, y1, '+', label='RK1')
    ax1.plot(x, y2, '+', label='RK2')
    ax1.plot(x, y4, '+', label='RK4')
    ax1.plot(x, y_exact, label='Exact')

    ax2.plot(x, abs(y1 - y_exact))
    ax2.plot(x, abs(y2 - y_exact))
    ax2.plot(x, abs(y4 - y_exact))

    ax1.legend()
    plt.show()


def compare_methods_2d():
    """
    Compare RK1, RK2 and RK4 methods on a harmonic oscillator problem:
    y'' + y = 0
    y(0) = 1
    y'(0) = 0
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)

    x = np.linspace(0, 10 * np.pi, 1000)
    y0 = np.array([1, 0])

    def f(y, t):
        return np.array([y[1], -y[0]])

    y_exact = np.cos(x)
    y1 = rk.rk_1(y0, x, f)[:, 0]
    y2 = rk.rk_2(y0, x, f)[:, 0]
    y4 = rk.rk_4(y0, x, f)[:, 0]

    ax1.plot(x, y1, '+', label='RK1')
    ax1.plot(x, y2, '+', label='RK2')
    ax1.plot(x, y4, '+', label='RK4')
    ax1.plot(x, y_exact, label='Exact')

    ax2.plot(x, abs(y1 - y_exact))
    ax2.plot(x, abs(y2 - y_exact))
    ax2.plot(x, abs(y4 - y_exact))

    ax1.legend()
    plt.show()


def test_bdf():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    # ax2 = fig.add_subplot(212, sharex=ax1)
    # ax2.grid(True)

    x = np.linspace(0, 10, 100)
    y0 = 1

    def f(y, t):
        return y * (np.sin(t) ** 2)

    y_exact = np.exp(x / 2 - np.sin(2 * x) / 4)
    y_6 = bdf.bdf_6(y0, x, f)

    ax1.plot(x, y_6, '+', label='BDF6')
    ax1.plot(x, y_exact, label='Exact')

    # ax2.plot(x, abs(y1 - y_exact))
    # ax2.plot(x, abs(y2 - y_exact))
    # ax2.plot(x, abs(y4 - y_exact))

    ax1.legend()
    plt.show()


if __name__ == '__main__':
    test_bdf()
