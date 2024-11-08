import numpy as np

__all__ = ['triu_to_symm']


def triu_to_symm(M: np.ndarray):
    M[np.tril_indices_from(M, k=-1)] = np.triu(M, k=1)
