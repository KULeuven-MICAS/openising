import numpy as np
from scipy.linalg import eigh

from ising.flow import LOGGER
from ising.stages.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm

def spectral_partitioning(model:IsingModel, nb_cores:int=2, sign=1) -> np.ndarray:
    if  nb_cores % 2 != 0:
        raise ValueError("Number of partitions must be even.")
    A = -triu_to_symm(model.J)
    D = np.diag(np.count_nonzero(A, axis=0))
    L = D - A

    _, V = eigh(L, D, subset_by_index=[0, 1])
    v = V[:, -1]
    mean = np.median(v)
    s = np.where(v >= mean, 1, -1)
    if nb_cores == 2:
        return s, v, mean
    else:
        model1 = IsingModel(model.J[s==1, :][:, s==1], model.h[s==1])
        model2 = IsingModel(model.J[s==-1, :][:, s==-1], model.h[s==-1])

        s1,_,_ = spectral_partitioning(model1, int(nb_cores / 2), sign=1)
        s2,_,_ = spectral_partitioning(model2, int(nb_cores / 2), sign=-1)
        if sign == -1:
            s[s==-1] = 4*s2
            s[s==1] = 3*s1
        else:
            s[s==-1] = 2*s2
            s[s==1] = s1
        return s, None, None

