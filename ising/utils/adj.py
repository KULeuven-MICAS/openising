import numpy as np


def complete(n: int) -> np.ndarray:
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    return adj


def king(n: int) -> np.ndarray:
    raise NotImplementedError()


def chimera(n: int) -> np.ndarray:
    raise NotImplementedError()
