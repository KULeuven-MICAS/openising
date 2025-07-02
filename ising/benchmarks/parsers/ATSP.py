import networkx as nx
import pathlib
import tsplib95

from ising.flow import TOP
from ising.utils.parser import get_optim_value

def ATSP_parser(benchmark:pathlib.Path|str) -> tuple[nx.DiGraph, float]:
    """Creates a networkx instance from the given benchmark. It is important that the benchmark originates from [here](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

    Args:
        benchmark (pathlib.Path | str): full path to the benchmark file

    Returns:
        G,best_found (tuple[nx.DiGraph, float]): a directed graph that originated from the given benchmark and the best
                                                 found solution energy.
    """
    if not benchmark.exists():
        raise FileNotFoundError(f"The benchmark file {benchmark} does not exist.")
    benchmark = str(benchmark)
    name = benchmark.split("/")[-1].split(".")[0]

    # Load the benchmark with the tsplib95 library.
    problem = tsplib95.load(benchmark)
    graph = problem.get_graph()
    graph.name = name
    best_found = get_optim_value(benchmark, TOP / "ising/benchmarks/ATSP/optimal_energy.txt")
    return graph, best_found
