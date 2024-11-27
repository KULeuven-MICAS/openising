import networkx as nx
import pathlib

def ATSP_parser(benchmark:pathlib.Path|str) -> nx.DiGraph:
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
    return G
