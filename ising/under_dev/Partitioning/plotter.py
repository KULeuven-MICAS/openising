import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from ising.flow import TOP
from ising.stages.model.ising import IsingModel
from ising.utils.helper_functions import make_directory

figtop = TOP / "ising/under_dev/Partitioning/plots"
make_directory(figtop)

def plot_partitioning(G:nx.Graph, s: list, fig_name: str):
    partitions = np.unique(s)
    communities = [[] for _ in range(len(np.unique(s)))]
    
    for ind, part in enumerate(s):
        part_ind = np.where(partitions == part)[0]
        communities[part_ind[0]].append(ind)
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes

    plt.figure(figsize=(50, 50))
    colors = plt.cm.tab10(np.arange(len(communities)))
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=community, node_color=colors[i])

    nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges)
    nx.draw_networkx_labels(G, pos=pos)
    plt.title("Partitioned graph")
    plt.savefig(figtop / fig_name)
    plt.close()

def plot_eigenvector(eigenvector:np.ndarray, mean:float, figname:str):
    sorted_eigenvector = np.sort(eigenvector)
    cut = np.where(sorted_eigenvector >= mean, 1, 0)
    n = len(eigenvector)
    plt.figure()
    plt.plot(sorted_eigenvector, color="k")
    plt.axhline(mean, linestyle="--", color="k")
    plt.ylabel("Eigenvector value")
    plt.title(f"Size of partition 1: {np.count_nonzero(cut)} out of {n} nodes")
    plt.savefig(figtop / figname, bbox_inches='tight')
    plt.close()

def plot_percentage_cut_edges(model:IsingModel, partitions_list:dict[str:np.ndarray], figname:str):
    total_edges = np.count_nonzero(model.J)
    cut_edges_per_partition = dict()
    nodes = np.arange(model.num_variables)
    for technique, partitioning in partitions_list.items():
        diff_partitions = np.unique(partitioning)
        cut_edges = 0
        for diff in diff_partitions:
            for other_diff in diff_partitions:
                if diff != other_diff:
                    # breakpoint()
                    overlapping_part = model.J[nodes[partitioning==diff], :][:, nodes[partitioning==other_diff]] # this is done to ensure mismatched sizes are still handled
                    cut_edges += np.count_nonzero(overlapping_part)
        cut_edges_per_partition[technique] = (cut_edges / 2)/total_edges*100
    plt.figure()
    plt.bar(cut_edges_per_partition.keys(), cut_edges_per_partition.values())
    plt.ylabel("Percentage of cut edges")
    plt.xlabel("Partitioning technique")
    plt.savefig(figtop / figname, bbox_inches='tight')
    plt.close()

def plot_energies_cores(cores:list[int], energies:list[float], best_energy:float,  figname:str):

    plt.figure()
    plt.plot(cores, energies, "x-")
    plt.axhline(best_energy, linestyle="--", color="k", label="Best energy")
    plt.xlabel("Number of cores")
    plt.ylabel("Optimal Energy")
    plt.legend()
    plt.savefig(figtop / figname, bbox_inches='tight')
    plt.close()
