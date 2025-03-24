import numpy as np

from ising.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm

def spectral_partitioning(model:IsingModel):
    A = triu_to_symm(model.J)
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    lam, V = np.linalg.eig(L)

    res = sorted(range(len(lam)), key=lambda sub: lam[sub])[:2]
    mean = np.median(V[:, res[-1]])
    s = np.where(V[:, res[-1]] >= mean, -1, 1)
    return s