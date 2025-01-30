import networkx as nx
import pathlib
import tsplib95
import os

def TSP_parser(benchmark:pathlib.Path)->tuple[nx.DiGraph, float]:
    """Creates a graph from the given benchmark. With this graph a TSP problem can be generated.
    It is important to note that only TSP benchmarks can be used

    Args:
        benchmark (pathlib.Path): the absolute path to the benchmark file.

    Returns:
        tuple[nx.DiGraph, float]: a tuple containing the graph and best found energy.
    """
    if not benchmark.exists():
        print("Benchmark does not exist")
        return None
    benchmark = str(benchmark)
    problem = tsplib95.load(benchmark)
    graph = problem.get_graph()
    best_found = get_optim_value(benchmark)
    return graph, best_found

def get_optim_value(benchmark:pathlib.Path):
    """Returns the best found energy of the given benchmark.

    Args:
        benchmark (pathlib.Path): the given benchmark.
    """
    benchmark = str(benchmark).split("/")[-1].split(".")[0]
    optim_file = pathlib.Path(os.getenv("TOP")) / pathlib.Path("ising/benchmarks/TSP/optimal_energy.txt")
    best_found = None

    with optim_file.open() as f:
        for line in f:
            line = line.split()
            if line[0] == benchmark:
                best_found = float(line[1])
                break

    return best_found
