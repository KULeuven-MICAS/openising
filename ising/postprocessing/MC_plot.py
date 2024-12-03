import networkx as nx
import matplotlib.pyplot as plt
import pathlib
import numpy as np

from ising.postprocessing.helper_functions import return_data

def plot_MC_solution(fileName:pathlib.Path, G_orig:nx.Graph, save:bool = True, save_folder:pathlib.Path = '.') -> None:
    """
    Plots the solution state of a Max-Cut problem.

    Args:
        fileName (pathlib.Path): the absolute path to the logfile of the optimisation process.
        G_orig (nx.Graph): the original graph of the problem.
        save (bool, optional): whether to save the plot. Defaults to True.
        save_folder (pathlib.Path, optional): the absolute path to the folder to save the plot in. Defaults to '.'.
    """
    G = nx.Graph()
    solutions_state = return_data(fileName=fileName, data="solution_state")
    solver = return_data(fileName=fileName, data="solver")
    best_energy = return_data(fileName=fileName, data="solution_energy")
    N = int(np.sqrt(np.shape(solutions_state)[0]))

    edges = []
    blue_nodes = set()
    red_nodes = set()
    for u in range(N):
        for v in range(u+1, N):
            if G_orig.has_edge(u, v) and solutions_state[u] != solutions_state[v]:
                edges.append((u, v))
                if solutions_state[u] == 1:
                    blue_nodes.add(u)
                    red_nodes.add(v)
                else:
                    blue_nodes.add(v)
                    red_nodes.add(u)

    G.add_nodes_from(list(blue_nodes))
    G.add_nodes_from(list(red_nodes))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=1)

    plt.figure()
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='b')
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=edges)
    plt.title(f"Solution state with optimal energy {best_energy}")
    if save:
        plt.savefig(f"{save_folder}/{solver}_MC_solution_state.png")
    plt.show()
