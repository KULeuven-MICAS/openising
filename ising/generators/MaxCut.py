import numpy as np
import networkx as nx
import pathlib
import os

from ising.model.ising import IsingModel
from ising.benchmarks.parsers.G import G_parser

TOP = pathlib.Path(os.getenv("TOP"))


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
        c -= weight / 2
    J = np.triu(J)
    return IsingModel(J, h, c, name=graph.name)


def random_MaxCut(N: int, seed:int=0) -> IsingModel:
    """Generates a random Max Cut problem. If there is a benchmark present for the given amount of nodes,
    the dummy benchmark is used.

    Args:
        N (int): size of the problem.

    Returns:
        IsingModel: generated problem.
    """
    filePath = TOP / f"ising/benchmarks/Maxcut_dummy/Dummy_N{N}.txt"
    if filePath.exists():
        graph, _ = G_parser(filePath)
        return MaxCut(graph)
    else:
        np.random.seed(seed)
        J = np.random.choice([-0.5, 0.0, 0.5], (N, N), p=[0.15, 0.7, 0.15])
        J = np.triu(J, k=1)
        h = np.zeros((N,))
        c = np.sum(J)
        return IsingModel(J, h, c)
