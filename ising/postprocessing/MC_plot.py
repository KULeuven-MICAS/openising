import networkx as nx
import matplotlib.pyplot as plt
import pathlib
import numpy as np

from ising.postprocessing.helper_functions import return_metadata

def plot_MC_solution(fileName:pathlib.Path, G_orig:nx.Graph, fig_name:str="MC_solution_state.png", save:bool = True,
                     save_folder:pathlib.Path = '.') -> None:
    """
    Plots the solution state of a Max-Cut problem.

    Args:
        fileName (pathlib.Path): the absolute path to the logfile of the optimisation process.
        G_orig (nx.Graph): the original graph of the problem.
        save (bool, optional): whether to save the plot. Defaults to True.
        save_folder (pathlib.Path, optional): the absolute path to the folder to save the plot in. Defaults to '.'.
    """
    G = nx.Graph()
    solutions_state = return_metadata(fileName=fileName, metadata="solution_state")
    best_energy = return_metadata(fileName=fileName, metadata="solution_energy")
    N = int(np.shape(solutions_state)[0])

    edges = []
    blue_nodes = set()
    red_nodes = set()
    labels = dict()
    for u in range(N):
        labels[u] = u+1
        if solutions_state[u] == 1:
            blue_nodes.add(u)
        else:
            red_nodes.add(u)

        for v in range(u+1, N):
            if G_orig.has_edge(u, v) and solutions_state[u] == solutions_state[v]:
                edges.append((u, v))

    G.add_nodes_from(G_orig.nodes(data=True))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=1)

    plt.figure()
    plt.subplot(1, 2, 2)
    nx.draw_networkx_nodes(G_orig, pos, nodelist=blue_nodes, node_color='b')
    nx.draw_networkx_nodes(G_orig, pos, nodelist=red_nodes, node_color='r')
    nx.draw_networkx_edges(G_orig, pos, edgelist=G_orig.edges(data=True))
    nx.draw_networkx_labels(G_orig, pos, labels)
    plt.title("Original graph")
    plt.subplot(1, 2, 1)
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='b')
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=edges)
    nx.draw_networkx_labels(G, pos, labels)
    plt.title(f"Solution state with optimal energy {best_energy}")
    if save:
        plt.savefig(f"{save_folder}/{fig_name}")
