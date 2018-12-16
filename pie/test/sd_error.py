import numpy as np
from pie.spatial import sd


def extrapolate_sd(method, y, n_extra=100):
    cell_extra = np.linspace(-1, 1, n_extra)
    x_extra = np.linspace(method.mesh[0], method.mesh[-1], method.n_cell * n_extra)

    # cell_extra = method.cell
    # n_extra = method.p
    # x_extra = method.x

    y_extra = np.zeros((method.n_cell * n_extra,))
    for i in range(method.n_cell):
        y_cell = y[i * method.p:(i + 1) * method.p]
        y_extra[i * n_extra:(i + 1) * n_extra] = sd.lagrange_extrapolation(method.cell, y_cell, cell_extra)

    return x_extra, y_extra


def sd_error(method, y1, y2, n_extra=100, type_error=0):
    _, y1 = extrapolate_sd(method, y1, n_extra=n_extra)
    _, y2 = extrapolate_sd(method, y2, n_extra=n_extra)
    if type_error == 1:
        return np.mean(abs(y1 - y2))
    elif type_error == 2:
        return np.sqrt(np.mean((y1 - y2) ** 2))
    else:
        return np.max(abs(y1 - y2))
