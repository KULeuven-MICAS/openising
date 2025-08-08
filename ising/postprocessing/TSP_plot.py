import matplotlib.pyplot as plt
import networkx as nx
import pathlib
import numpy as np

from ising.postprocessing.helper_functions import return_metadata


def plot_graph_solution(
    fileName: pathlib.Path, G_orig: nx.DiGraph, figName: str, save: bool = True, save_folder: pathlib.Path = "."
):
    """Plots the solution state of a TSP problem.

    Args:
        fileName (pathlib.Path): absolute path to the logfile of the optimisation process.
        G_orig (nx.DiGraph): original graph of the TSP problem.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to '.'.
    """
    # G = nx.DiGraph()
    solution_state = return_metadata(fileName=fileName, metadata="solution_state")
    best_energy = return_metadata(fileName=fileName, metadata="solution_TSP_energy")
    N = int(np.sqrt(np.shape(solution_state)[0]))

    red_edges = []
    nodes_in_path = []
    path = [None] * N
    sub_state = solution_state.reshape((N, N))
    for city in range(N):
        city_state = sub_state[:, city]
        if np.any(city_state == 1):
            nodes_in_path.append(city + 1)
        ind = np.where(city_state == 1)[0]
        if ind.size != 0:
            path[ind[0]] = city + 1

    first_city = -1
    for i in range(N):
        city1 = path[i]
        if i == 0:
            first_city = city1
        city2 = path[(i + 1) if i + 1 < N else 0]
        if G_orig.has_edge(city1, city2):
            red_edges.append((city1, city2))
            red_edges.append((city2, city1))
    black_edges = [(i, j) for (i, j) in G_orig.edges() if (i, j) not in red_edges and i != j]
    pos = nx.spring_layout(G_orig, k=5 / np.sqrt(G_orig.order()), seed=1)

    plt.figure()
    nx.draw_networkx_nodes(G_orig, pos, nodelist=list(range(1, N + 1)), node_color="b")
    nx.draw_networkx_nodes(G_orig, pos, nodelist=nodes_in_path, node_color="r")
    if first_city != -1:
        nx.draw_networkx_nodes(G_orig, pos, nodelist=[first_city], node_color="g")
    nx.draw_networkx_edges(G_orig, pos, edgelist=red_edges, edge_color="r", connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G_orig, pos, edgelist=black_edges, edge_color="k", connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G_orig, pos, labels={i: i for i in range(1, N + 1)})
    plt.title(f"Solution state with optimal energy {best_energy}")
    if save:
        plt.savefig(save_folder / f"{figName}.pdf")
    plt.close()
