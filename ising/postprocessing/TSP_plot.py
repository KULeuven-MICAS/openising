import matplotlib.pyplot as plt
import networkx as nx
import pathlib
import numpy as np

from ising.postprocessing.helper_functions import return_data

def plot_graph_solution(fileName:pathlib.Path, save:bool=True, save_folder:pathlib.Path='.'):
    G = nx.DiGraph()
    solutions_state = return_data(fileName=fileName, data="solution_state")
    solver = return_data(fileName=fileName, data="solver")
    best_energy = return_data(fileName=fileName, data="solution_energy")
    N = int(np.sqrt(np.shape(solutions_state)[0]))

    G.add_nodes_from(list(range(N)))
    u = None
    v = None
    edges = []
    for time in range(N):
        for city in range(N):
            if solutions_state[city*N + time] == 1:
                if u is None:
                    u = city
                else:
                    v = city
                    edges.append((u,v))
                    u = v
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=1)

    plt.figure()
    nx.draw_networkx_nodes(G, pos, nodelist=list(range(N)))
    nx.draw_networkx_edges(G, pos, edgelist=edges)
    plt.title(f"Solution state with optimal energy {best_energy}")
    if save:
        plt.savefig(f"{save_folder}/{solver}_solution_state.png")
    plt.show()
