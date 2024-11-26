import pathlib
import networkx as nx


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
                row = list(line)
                N = ""
                for i in range(len(row)):
                    if row[i] == " " and len(N) > 0:
                        N = int(N)
                        break
                    elif row[i] != " ":
                        N += row[i]
                G.add_nodes_from(list(range(N)))
                data = True
            else:
                line = list(line)
                first_node = True
                number = ""
                for i in range(len(line)):
                    if line[i] == " " and len(number) > 0:
                        if first_node:
                            u = int(number) - 1
                            first_node = False
                        else:
                            v = int(number) - 1
                        number = ""
                    elif line[i] != " ":
                        number += line[i]
                weight = int(number)
                G.add_edge(u, v, weight=weight)
    return G
