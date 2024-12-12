import numpy as np
from ising.model.ising import IsingModel
import networkx as nx


def MaxCut(graph: nx.Graph) -> IsingModel:
    """Generates an Ising model from the given undirected graph

    Args:
        graph (nx.Graph): graph on which the max-cut problem will be solved

    Returns:
        model (IsingModel): generated model from the graph
    """
    N = len(graph.nodes)
    J = np.zeros((N, N))
    h = np.zeros((N,))
    c = 0.0
    for node1, node2 in graph.edges:
        weight = graph[node1][node2]["weight"]
        J[node1, node2] = -weight / 2
        J[node2, node1] = -weight / 2
        c += weight
    J = np.triu(J)
    return IsingModel(J, h, -1 / 2 * c)


def random_MaxCut(N: int) -> IsingModel:
    """Generates a random Max Cut problem.

    Args:
        N (int): size of the problem.

    Returns:
        IsingModel: generated problem.
    """
    J = np.random.choice([-0.5,0., 0.5], (N,N))
    J = np.triu(J, k=1)
    h = np.zeros((N,))
    c = np.sum(J)
    return IsingModel(J, h, c)
