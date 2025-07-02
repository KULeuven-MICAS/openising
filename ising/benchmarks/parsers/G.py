import pathlib
import networkx as nx

from ising.flow import TOP
from ising.utils.parser import get_optim_value

def G_parser(benchmark: pathlib.Path | str):
    """Creates undirected graph from G benchmark. All the benchmark files should be in the same format, which is:

    ``node_i node_j weight_ij``

    And the first line holds the amount of nodes and edges in the problem.

    Args:
        benchmark (pathlib.Path | str): benchmark that needs to be generated

    Returns:
        G,best_found (tuple[nx.Graph, float]): graph generated from the benchmark and the best found cut value.
    """
    # Make sure we keep track of where we are in the file
    data = False
    name = str(benchmark).split("/")[-1].split(".")[0]

    # Initialize the graph
    G = nx.Graph(name=name)
    with benchmark.open() as f:
        for line in f:
            if not data:
                # When the data is not yet read, we are at the first line of the file.
                # We can read the amount of nodes and put the data part to True.
                row = line.split()
                N = int(row[0])
                G.add_nodes_from(list(range(N)))
                data = True
            else:
                # Every line holds the first node, the second node and the weight of the edge.
                line = line.split()
                u = int(line[0]) - 1
                v = int(line[1]) - 1
                weight = int(line[2])
                G.add_edge(u, v, weight=weight)

    best_found = get_optim_value(benchmark, TOP / "ising/benchmarks/G/optimal_energy.txt")
    return G, best_found

