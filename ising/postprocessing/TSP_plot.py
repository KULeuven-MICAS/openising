import matplotlib.pyplot as plt
import networkx as nx
import pathlib
import numpy as np

from ising.postprocessing.helper_functions import return_metadata
from ising.generators.TSP import get_index


def plot_graph_solution(
    fileName: pathlib.Path, G_orig: nx.DiGraph, fig_name: str, save: bool = True, save_folder: pathlib.Path = "."
):
    """Plots the solution state of a TSP problem.

    Args:
        fileName (pathlib.Path): absolute path to the logfile of the optimisation process.
        G_orig (nx.DiGraph): original graph of the TSP problem.
        save (bool, optional): whether to save the figure. Defaults to True.
        save_folder (pathlib.Path, optional): where to save the figure. Defaults to '.'.
    """
    # G = nx.DiGraph()
    solutions_state = return_metadata(fileName=fileName, metadata="solution_state")
    best_energy = return_metadata(fileName=fileName, metadata="solution_TSP_energy")
    N = int(np.sqrt(np.shape(solutions_state)[0]))

    red_edges = []
    path = [-1]*N
    for time in range(N):
        for city in range(N):
            index = get_index(time, city, N)
            if solutions_state[index] == 1:
                path[time] = city
                break

    for i in range(N):
        city1 = path[i]
        city2 = path[(i+1) % N]
        if G_orig.has_edge(city1, city2):
            red_edges.append((city1, city2))
    black_edges = [(i, j) for (i, j) in G_orig.edges() if (i, j) not in red_edges]

    pos = nx.spring_layout(G_orig, k=5/np.sqrt(G_orig.order()), seed=1)

    plt.figure()
    nx.draw_networkx_nodes(G_orig, pos, nodelist=list(range(N)), node_color="b")
    nx.draw_networkx_edges(G_orig, pos, edgelist=red_edges, edge_color="r")
    nx.draw_networkx_edges(G_orig, pos, edgelist=black_edges, edge_color="k")
    nx.draw_networkx_labels(G_orig, pos, labels={i: i+1 for i in range(N)})
    plt.title(f"Solution state with optimal energy {best_energy}")
    if save:
        plt.savefig(f"{save_folder}/{fig_name}")
    plt.close()
