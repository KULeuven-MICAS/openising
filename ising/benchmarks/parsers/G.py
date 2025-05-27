import pathlib
import networkx as nx

from ising.flow import TOP
from ising.utils.parser import get_optim_value

def G_parser(benchmark: pathlib.Path | str):
    """Creates undirected graph from G benchmark

    Args:
        benchmark (pathlib.Path | str): benchmark that needs to be generated

    Returns:
        G (nx.Graph): graph generated from the benchmark
        best_found (float): best found cut value.
    """
    data = False
    name = str(benchmark).split("/")[-1].split(".")[0]
    G = nx.Graph(name=name)
    with benchmark.open() as f:
        for line in f:
            if not data:
                row = line.split()
                N = int(row[0])
                G.add_nodes_from(list(range(N)))
                data = True
            else:
                line = line.split()
                u = int(line[0]) - 1
                v = int(line[1]) - 1
                weight = int(line[2])
                G.add_edge(u, v, weight=weight)

    best_found = get_optim_value(benchmark, TOP / "ising/benchmarks/G/optimal_energy.txt")
    return G, best_found

