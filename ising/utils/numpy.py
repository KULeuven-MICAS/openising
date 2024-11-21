import numpy as np


def triu_to_symm(m: np.ndarray, inplace: bool = False):
    if inplace:
        m += np.triu(m, k=1).T
        return m
    else:
        return m + np.triu(m, k=1).T


def is_square(m: np.ndarray):
    return m.ndim == 2 and m.shape[0] == m.shape[1]


def is_triu(m: np.ndarray, k: int = 0):
    mask = np.tril(np.ones_like(m, dtype=bool), k=k - 1)
    return np.allclose(m[mask], 0)


def is_symmetric(m: np.ndarray):
    return np.allclose(m, m.T)


def is_diagonal(m: np.ndarray):
    return np.allclose(m, np.diag(np.diagonal(m)))


def has_zero_diagonal(m: np.ndarray):
    return np.allclose(np.diagonal(m), 0)
