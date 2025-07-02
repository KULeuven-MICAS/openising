import networkx as nx
import pathlib
import tsplib95

from ising.flow import TOP
from ising.utils.parser import get_optim_value

def TSP_parser(benchmark:pathlib.Path)->tuple[nx.DiGraph, float]:
    """Creates a graph from the given benchmark. With this graph a TSP problem can be generated.
    It is important to note that only TSP benchmarks can be used from [here](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

    Args:
        benchmark (pathlib.Path): the absolute path to the benchmark file.

    Returns:
        graph,best_found tuple[nx.DiGraph, float]: a tuple containing the graph and best found energy.
    """
    # Ensure the benchmark exists
    if not benchmark.exists():
        raise FileNotFoundError(f"The benchmark file {benchmark} does not exist.")
    benchmark = str(benchmark)

    # Benchmark is loaded using tsplib95, that efficiently parses the benchmark file.
    problem = tsplib95.load(benchmark)
    graph = problem.get_graph()
    graph.name = benchmark.split("/")[-1].split(".")[0]
    best_found = get_optim_value(benchmark, TOP / "ising/benchmarks/TSP/optimal_energy.txt")
    return graph, best_found

