import numpy as np

from ising.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm

def spectral_partitioning(model:IsingModel):
    J = triu_to_symm(model.J)
    A = np.where(J != 0, 1.0, 0.)
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    lam, V = np.linalg.eig(L)

    res = sorted(range(len(lam)), key=lambda sub: lam[sub])[:2]
    s = np.sign(V[:, res[1]])
    return s