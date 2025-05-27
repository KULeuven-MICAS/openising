import networkx as nx
import pathlib
import tsplib95

from ising.flow import TOP
from ising.utils.parser import get_optim_value

def ATSP_parser(benchmark:pathlib.Path|str) -> tuple[nx.DiGraph, float]:
    """creates a networkx instance fromthe given benchmark.

    Args:
        benchmark (pathlib.Path | str): full path to the benchmark file

    Returns:
        G (nx.DiGraph): a directed graph that originated from the given benchmark.
    """
    if not benchmark.exists():
        print("Benchmark does not exist")
        return None
    benchmark = str(benchmark)
    name = benchmark.split("/")[-1].split(".")[0]
    problem = tsplib95.load(benchmark)
    G = problem.get_graph()
    G.name = name
    best_found = get_optim_value(benchmark, TOP / "ising/benchmarks/ATSP/optimal_energy.txt")
    return G, best_found
