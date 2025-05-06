import networkx as nx
import pathlib
import os
import tsplib95

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
    best_found = get_optim_value(benchmark)
    return G, best_found


def get_optim_value(benchmark:pathlib.Path|str)->float:
    """Returns the best found energy of the benchmark.

    Args:
        benchmark (pathlib.Path | str): the benchmark file

    Returns:
        float: the best found energy of the benchmark
    """
    benchmark = str(benchmark).split("/")[-1].split(".")[0]
    optim_file = pathlib.Path(os.getenv("TOP")) / pathlib.Path("ising/benchmarks/ATSP/optimal_energy.txt")
    best_found = None

    with optim_file.open() as f:
        for line in f:
            line = line.split()
            if line[0] == benchmark:
                best_found = float(line[1])
                break

    return best_found
