import networkx as nx
import pathlib
import os

def ATSP_parser(benchmark:pathlib.Path|str) -> tuple[nx.DiGraph, float]:
    """creates a networkx instance fromthe given benchmark.

    Args:
        benchmark (pathlib.Path | str): full path to the benchmark file

    Returns:
        G (nx.DiGraph): a directed graph that originated from the given benchmark.
    """
    last_weight_in_row = 0
    row = 0
    N = int
    G = nx.DiGraph()
    weight_section = False
    with benchmark.open() as f:
        for line in f:
            if line[:9] == "DIMENSION":
                N = int(line[10:])
                G.add_nodes_from(list(range(N)))
            elif line[:19] == "EDGE_WEIGHT_SECTION":
                weight_section = True
            elif weight_section:
                line = list(line)
                part_row = []
                number = ""
                for i in range(len(line)):
                    if line[i] == " " and len(number) > 0:
                        part_row.append(int(number))
                        number = ""
                    elif line[i] != " ":
                        number += line[i]
                part_row.append(int(number))
                n = len(part_row)
                for other_node in range(last_weight_in_row, last_weight_in_row + n):
                    if part_row[other_node-last_weight_in_row] != 0.:
                        G.add_edge(row, other_node, weight=part_row[other_node-last_weight_in_row])
                last_weight_in_row += n
                if last_weight_in_row >= N:
                    last_weight_in_row = 0
                    row += 1
                if row == N:
                    weight_section = False
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
