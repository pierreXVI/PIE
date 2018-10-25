import matplotlib.pyplot as plt
import numpy as np
import rk
import bdf


def f(y, t):
    return y * (np.sin(t) ** 2)


def solution(t):
    return np.exp(t / 2 - np.sin(2 * t) / 4)


def compare_methods_rk():
    """
    Compare RK1, RK2 and RK4 methods
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)

    x = np.linspace(0, 10, 100)
    y0 = 1

    y_exact = solution(x)
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


def compare_methods_bdf():
    """
    Compare BDF1-6 methods
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    x = np.linspace(0, 10, 100)
    y0 = 1

    y_exact = solution(x)
    ax1.plot(x, y_exact, c='k', label=r'$y_{exact}$')

    for method, label in zip((bdf.bdf_1, bdf.bdf_2, bdf.bdf_3, bdf.bdf_4, bdf.bdf_5, bdf.bdf_6),
                             ('BDF1', 'BDF2', 'BDF3', 'BDF4', 'BDF5', 'BDF6')):
        y = method(y0, x, f)
        color = ax1.plot(x, y, '+', label=label)[0].get_color()
        ax2.semilogy(x, abs(y - y_exact), c=color)

    ax1.legend()
    plt.show()


def compare_all_methods():
    """
    Compare all methods
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    x = np.linspace(0, 10, 100)
    y0 = 1

    y_exact = solution(x)
    ax1.plot(x, y_exact, c='k', label=r'$y_{exact}$')

    score = dict()
    for method, label in zip(
            (rk.rk_1, rk.rk_2, rk.rk_4, bdf.bdf_1, bdf.bdf_2, bdf.bdf_3, bdf.bdf_4, bdf.bdf_5, bdf.bdf_6),
            ('RK1', 'RK2', 'RK4', 'BDF1', 'BDF2', 'BDF3', 'BDF4', 'BDF5', 'BDF6')):
        y = method(y0, x, f)
        color = ax1.plot(x, y, '+', label=label)[0].get_color()
        ax2.semilogy(x, abs(y - y_exact), c=color)
        score[label] = np.sum(abs(y - y_exact))

    print('Scoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend()
    plt.show()


def compare_methods_2d():
    """
    Compare methods on a harmonic oscillator problem:
    y'' + y = 0
    y(0) = 1
    y'(0) = 0
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.set_ylabel(r'$y$')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$\left|y - y_{exact}\right|$')

    x = np.linspace(0, 10 * np.pi, 1000)
    y0 = np.array([1, 0])

    def f_harmonic(y, t):
        return np.array([y[1], -y[0]])

    y_exact = np.cos(x)
    ax1.plot(x, y_exact, c='k', label=r'$y_{exact}$')

    score = dict()
    for method, label in zip(
            (rk.rk_1, rk.rk_2, rk.rk_4, bdf.bdf_1, bdf.bdf_2, bdf.bdf_3, bdf.bdf_4, bdf.bdf_5, bdf.bdf_6),
            ('RK1', 'RK2', 'RK4', 'BDF1', 'BDF2', 'BDF3', 'BDF4', 'BDF5', 'BDF6')):
        y = method(y0, x, f_harmonic)
        color = ax1.plot(x, y[:, 0], '+', label=label)[0].get_color()
        ax2.semilogy(x, abs(y[:, 0] - y_exact), c=color)
        score[label] = np.sum(abs(y[:, 0] - y_exact))

    print('Scoring :')
    print(*sorted(score, key=score.__getitem__), sep=' > ')

    ax1.legend()
    plt.show()


if __name__ == '__main__':
    # compare_methods_rk()
    compare_methods_bdf()
    compare_all_methods()
    compare_methods_2d()
