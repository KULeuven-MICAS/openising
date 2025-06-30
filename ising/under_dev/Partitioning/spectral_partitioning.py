import numpy as np
from scipy.linalg import eigh

from ising.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm

def spectral_partitioning(model:IsingModel, nb_partitions:int=2) -> np.ndarray:
    if  nb_partitions % 2 != 0:
        raise ValueError("Number of partitions must be even.")
    A = triu_to_symm(model.J)
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    _, V = eigh(L, D, subset_by_index=[0, 1])

    mean = np.median(V[:, -1])
    s = np.where(V[:, -1] >= mean, 1, -1)
    if nb_partitions == 2:
        return s
    else:
        model1 = IsingModel(model.J[s==1, s==1], model.h[s==1])
        model2 = IsingModel(model.J[s==-1, s==-1], model.h[s==-1])

        s1 = spectral_partitioning(model1, nb_partitions / 2)
        s2 = 2*spectral_partitioning(model2, nb_partitions / 2)
        s[s==1] = s1
        s[s==-1] = s2
        return s

