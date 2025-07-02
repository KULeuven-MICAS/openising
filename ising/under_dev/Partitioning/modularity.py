import numpy as np
from scipy.linalg import eigh

from ising.stages.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm


def partitioning_modularity(model:IsingModel, nb_cores:int=2):
    """partitions the models nodes into a number of partitions according to the modularity partitioning scheme.
    The number of partitions is determined by the parameter nb_cores.

    Args:
        model (IsingModel): the model to be partitioned
        nb_cores (int): the number of partitions to be created

    Returns:
        s (np.ndarray): the partitioning of the model 
    """
    if nb_cores % 2 != 0:
        raise ValueError("Only even amount of cores are allowed")
    n = model.num_variables
    A = triu_to_symm(model.J)
    k = np.count_nonzero(A, axis=0)
    m = np.sum(k) / 2
    B = A - np.outer(k, k) / (2*m)
    D = np.diag(k)

    _, V = eigh(B, D, subset_by_index=[n-2, n-1])
    v = V[:, -1]
    mean = np.mean(v)
    s = np.where(v >= mean, 1, -1)

    if len(np.unique(s)) == 1:
        v = V[:, -2]
        s = np.sign(v)

    if nb_cores == 2:
        return s, v, mean

    else:
        model1 = IsingModel(model.J[s==1, s==1], model.h[s==1])
        model2 = IsingModel(model.J[s==-1, s==-1], model.h[s==-1])

        s1, v1 = partitioning_modularity(model1, nb_cores / 2)
        s2, v2 = partitioning_modularity(model2, nb_cores / 2)
        s[s==1] = s1
        s[s==-1] = 2*s2
        return s, None