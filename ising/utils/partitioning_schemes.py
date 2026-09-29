import numpy as np
from scipy.linalg import eigh
import networkx as nx

from ising.stages.model.ising import IsingModel
from ising.utils.numpy import triu_to_symm


def make_random_partitioning(model: IsingModel, nb_cores: int) -> np.ndarray:
    """Return balanced, nonempty random partitions in original-node order."""
    if not 1 <= nb_cores <= model.num_variables:
        raise ValueError("Partition count must be between one and the number of nodes")
    labels = np.empty(model.num_variables, dtype=int)
    for part_id, nodes in enumerate(np.array_split(np.random.permutation(model.num_variables), nb_cores)):
        labels[nodes] = part_id
    return labels

def make_modularity_partitioning(model: IsingModel, nb_cores: int) -> np.ndarray:
    return partition_by_eigenvector(model, nb_cores, spectral=False)

def make_spectral_partitioning( model: IsingModel, nb_cores: int) -> np.ndarray:
    return partition_by_eigenvector(model, nb_cores, spectral=True)

def make_louvain_partitioning(model: IsingModel, nb_cores:int) -> np.ndarray:
    G = nx.Graph(model.J)
    partitions = nx.algorithms.community.louvain_communities(G, max_level=int(nb_cores/2))
    labels = np.empty(model.num_variables, dtype=int)
    for label, partition in enumerate(partitions):
        labels[list(partition)] = label
    return labels

def partition_by_eigenvector(model: IsingModel, nb_cores: int, *, spectral: bool) -> np.ndarray:
    """Bisect by rank, keeping labels distinct across branches.

    Keep the existing generalized eigenproblems. Their degree matrix must
    be positive definite, as required by scipy.linalg.eigh.
    """
    n = model.num_variables
    if not 1 <= nb_cores <= n or nb_cores & (nb_cores - 1):
        raise ValueError(
            "Eigenvector partitioning requires a power-of-two partition count no larger than the model"
        )
    if nb_cores == 1:
        return np.zeros(n, dtype=int)
    if nb_cores == n:
        return np.arange(n)

    adjacency = triu_to_symm(model.J)
    if spectral:
        degree = np.diag(np.sum(adjacency, axis=0))
        _, vectors = eigh(degree - adjacency, degree, subset_by_index=[1, 1])
    else:
        degree = np.diag(np.count_nonzero(adjacency, axis=0))
        _, vectors = eigh(adjacency, degree, subset_by_index=[n - 2, n - 2])
    order = np.argsort(vectors[:, 0], kind="stable")
    labels = np.empty(n, dtype=int)
    for branch, nodes in enumerate(np.array_split(order, 2)):
        # Preserve the upper-triangular J convention when extracting submodels.
        nodes = np.sort(nodes)
        local_model = IsingModel(model.J[np.ix_(nodes, nodes)], model.h[nodes])
        labels[nodes] = partition_by_eigenvector(local_model, nb_cores // 2, spectral=spectral)
        labels[nodes] += branch * (nb_cores // 2)
    return labels
