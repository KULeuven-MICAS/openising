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
    s = do_modularity(model)
    if nb_cores == 2:
        return s

    elif nb_cores == 3:
        _, model_n = get_partition_of_model(model, s)
        s_n = partitioning_modularity(model_n, nb_cores=2)
        s[s==-1] = 2*s_n

        return s
    elif nb_cores % 2 != 0:
        model_p, model_n = get_partition_of_model(model, s)
        s_p = partitioning_modularity(model_p, nb_cores = nb_cores/2-0.5)
        s_n = partitioning_modularity(model_n, nb_cores = nb_cores/2+0.5)
        s[s==-1] = 2*s_n
        s[s==1] = s_p
        return s
    else:
        model_p, model_n = get_partition_of_model(model, s)
        s_p = partitioning_modularity(model_p, nb_cores = nb_cores/2)
        s_n = partitioning_modularity(model_n, nb_cores = nb_cores/2)
        s[s==-1] = 2*s_n
        s[s==1] = s_p
        return s


def do_modularity(model:IsingModel):
    J = triu_to_symm(model.J)
    A = np.where(J != 0, 1., 0.)
    k = np.sum(A, axis=0)
    m = np.sum(k) / 2
    B = A - np.outer(k, k) / (2*m)

    lam, V = np.linalg.eig(B)
    v = V[:,np.argmax(lam)]
    s = np.sign(v)

    return s

def get_partition_of_model(model: IsingModel, s:np.ndarray):
    Jp = np.zeros((len(s[s==1]), len(s[s==1])))
    Jn = np.zeros((len(s[s==1]), len(s[s==1])))

    Jp = model.J[s==1, :]
    Jp = Jp[:, s==1]
    Jn = model.J[s==-1, :]
    Jn = Jn[:, s==-1]

    return IsingModel(Jp, np.zeros(len(s[s==1]))), IsingModel(Jn, np.zeros(len(s[s==-1])))