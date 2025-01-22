import pathlib
import networkx as nx
import os

def G_parser(benchmark: pathlib.Path | str):
    """Creates undirected graph from G benchmark

    Args:
        benchmark (pathlib.Path | str): benchmark that needs to be generated

    Returns:
        G (nx.Graph): graph generated from the benchmark
    """
    data = False
    G = nx.Graph()
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

    best_found = get_optim_value(benchmark)
    return G, best_found

def get_optim_value(benchmark: pathlib.Path | str) -> float:
    """Returns the best found value of the benchmark if the optimal value is known.

    Args:
        benchmark (pathlib.Path | str): absolute path to the benchmark

    Returns:
        float: the best found value of the benchmark.
    """
    best_found = None
    optim_file = pathlib.Path(os.getenv("TOP")) / pathlib.Path("ising/benchmarks/G/optimal_energy.txt")
    benchmark = str(benchmark).split("/")[-1][:-4]
    with optim_file.open() as f:
        for line in f:
            line = line.split()
            if line[0] == benchmark:
                best_found = int(line[1])
                break
    return best_found
