import numpy as np

from ising.model.ising import IsingModel
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
    J = triu_to_symm(model.J)
    A = np.where(J != 0, 1., 0.)
    k = np.sum(A, axis=0)
    m = np.sum(k) / 2
    B = A - np.outer(k, k) / (2*m)

    lam, V = np.linalg.eig(B)
    v = V[:,np.argmax(lam)]
    
    # Calculate thresholds using percentiles
    thresholds = np.percentile(v, np.linspace(0, 100, nb_cores+1)[1:-1])
    # Initialize partitioning array
    s = np.zeros(model.num_variables)
    
    # Assign partitions based on thresholds
    for i in range(1, nb_cores):
        if i == nb_cores - 1:
            mask = v >= thresholds[-1]
        else:
            mask = (v >= thresholds[i-1]) & (v < thresholds[i])
        s[mask] = i
        print(s)
    return s

