import numpy as np
import scipy.linalg


def phi_1(z):
    """
    The `phi_1` function as defined in "A short course on exponential integrators" by Marlis Hochbruck
    phi_1(z) = \int_0^1 \e^{\left(1 - s\right)z}\mathrm{d}s

    If z is float_like, phi_1(z) = (exp(z) - 1)/z
    If z is invertible, phi_1(z) = (exp(z) - Id) . z^-1
    Else, phi_1(z) = \sum_{n \gt 0} z^n/(n + 1)!

    :param z: array_like
    :return: phi_1(z)
    """
    if type(z) == np.ndarray:
        try:
            return np.dot(scipy.linalg.expm(z) - np.eye(z.shape[0]), np.linalg.inv(z))
        except np.linalg.LinAlgError:
            return _phi_1_approx(z)
    return (np.exp(z) - 1) / z


def _phi_1_approx(z, eps=1E-12):
    i = 1
    u = np.eye(z.shape[0])
    s = np.eye(z.shape[0])
    while True:
        i += 1
        u = np.dot(u, z / i)
        s = s + u

        if np.linalg.norm(u) / np.linalg.norm(s) < eps:
            return s


def test_phi_1(eps=1E-10):
    def err(foo, bar):
        if abs(bar) < eps:
            return abs(foo)
        return 100 * abs(1 - foo / bar)

    err = np.vectorize(err)

    phi1_1 = 1.718281828459045235360287471352662497757247093699959574966
    phi1_10 = 2202.546579480671651695790064528424436635351261855678107423
    exp_10 = 22026.46579480671651695790064528424436635351261855678107423
    phi1_100 = 2.6881171418161354484126255515800135873611117773741922E41
    exp_100 = 2.6881171418161354484126255515800135873611118773741922E43

    flag = True
    print('Test scalar')
    flag = flag and err(phi_1(1), phi1_1) < eps
    flag = flag and err(phi_1(10), phi1_10) < eps
    flag = flag and err(phi_1(100), phi1_100) < eps
    print('\t' + flag * 'Success' + (not flag) * 'Failure')

    flag = True
    print('Test invertible matrix')
    i_2 = np.eye(2)
    flag = flag and (err(phi_1(i_2), phi1_1 * i_2) < eps).all()
    flag = flag and (err(phi_1(10 * i_2), phi1_10 * i_2) < eps).all()
    flag = flag and (err(phi_1(100 * i_2), phi1_100 * i_2) < eps).all()

    a = np.array([[1, 1], [0, 1]])
    phi1_a = np.array([[phi1_1, 1], [0, phi1_1]])
    flag = flag and (err(phi_1(a), phi1_a) < eps).all()
    phi1_10a = np.array([[phi1_10, exp_10 - phi1_10], [0, phi1_10]])
    flag = flag and (err(phi_1(10 * a), phi1_10a) < eps).all()
    phi1_100a = np.array([[phi1_100, exp_100 - phi1_100], [0, phi1_100]])
    flag = flag and (err(phi_1(100 * a), phi1_100a) < eps).all()
    print('\t' + flag * 'Success' + (not flag) * 'Failure')

    flag = True
    print('Test non invertible matrix')
    b = np.array([[1, 0], [0, 0]])
    phi1_b = np.array([[phi1_1, 0], [0, 1]])
    flag = flag and (err(phi_1(b), phi1_b) < eps).all()
    phi1_10b = np.array([[phi1_10, 0], [0, 1]])
    flag = flag and (err(phi_1(10 * b), phi1_10b) < eps).all()
    phi1_100b = np.array([[phi1_100, 0], [0, 1]])
    flag = flag and (err(phi_1(100 * b), phi1_100b) < eps).all()
    print('\t' + flag * 'Success' + (not flag) * 'Failure')

    flag = True
    print('Test non well conditioned')
    c = np.array([[1, 1000], [0, 100]])
    phi1_c = np.array([[phi1_1, 1000 * (phi1_100 - phi1_1) / 99], [0, phi1_100]])
    flag = flag and (err(phi_1(c), phi1_c) < eps).all()
    d = np.array([[1, 99], [1, 99]])
    phi1_d = phi1_100 * d / 100
    flag = flag and (err(phi_1(d), phi1_d) < eps).all()
    print('\t' + flag * 'Success' + (not flag) * 'Failure')


if __name__ == '__main__':
    test_phi_1()
