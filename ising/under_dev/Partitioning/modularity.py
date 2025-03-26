import numpy as np

from ising.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm


def partitioning_modularity(model:IsingModel):
    J = triu_to_symm(model.J)
    A = np.where(J != 0, 1., 0.)
    k = np.sum(A, axis=0)
    m = np.sum(k) / 2
    B = A - np.outer(k, k) / (2*m)

    lam, V = np.linalg.eig(B)
    s = np.sign(V[:,np.argmax(lam)])
    s = np.where(s == 0, -1, s)
    return s

