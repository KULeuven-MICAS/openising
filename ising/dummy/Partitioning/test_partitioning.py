import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pathlib

from ising.generators.MaxCut import MaxCut
from ising.dummy.Partitioning.modularity import partitioning_modularity
from ising.dummy.Partitioning.spectral_partitioning import spectral_partitioning
from ising.dummy.Partitioning.random_partitioning import random_partitioning

from ising.utils.flow import make_directory

TOP = pathlib.Path(os.getenv("TOP"))
figtop = TOP / 'ising/dummy/Partitioning/plots'
make_directory(figtop)

def plot_partitioning(G, s, fig_name):
    pos = nx.spring_layout(G, k=5/np.sqrt(5))
    _, (ax1, ax2) = plt.subplots(2, 1)

    nx.draw_networkx(G, pos, ax=ax1)

    red_nodes = [node for node in G.nodes if s[node]==1.]
    blue_nodes = [node for node in G.nodes if s[node]==-1.]
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='tab:red', ax=ax2)
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='tab:blue', ax=ax2)
    for edge in G.edges:
        if s[edge[0]] != s[edge[1]]:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax2, style='--')
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax2)
    nx.draw_networkx_labels(G, pos, ax=ax2)
    
    plt.savefig(figtop / fig_name)


def test_compare_small():
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edge(0, 2, weight=1)
    G.add_edge(2, 3, weight=-1)
    G.add_edge(3, 1, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(1, 4, weight=1)

    model = MaxCut(G)
    s_mod = partitioning_modularity(model)
    s_spec = spectral_partitioning(model)
    s_rand = random_partitioning(model)
    plot_partitioning(G, s_mod, "modularity_test.png")
    plot_partitioning(G, s_spec, "spectral_test.png")
    plot_partitioning(G, s_rand, "random_test.png")


def test_compare_big():
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edge(0, 3, weight=1)
    G.add_edge(0, 6, weight=1)
    G.add_edge(1, 4, weight=1)
    G.add_edge(1, 5, weight=1)
    G.add_edge(1, 9, weight=1)
    G.add_edge(2, 5, weight=1)
    G.add_edge(2, 7, weight=1)
    G.add_edge(2, 9, weight=1)
    G.add_edge(3, 5, weight=1)
    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 7, weight=1)
    G.add_edge(6, 9, weight=1)
    G.add_edge(6, 7, weight=1)
    G.add_edge(8, 9, weight=1)

    model = MaxCut(G)
    s_mod = partitioning_modularity(model)
    s_spec = spectral_partitioning(model)
    s_rand = random_partitioning(model)
    plot_partitioning(G, s_mod, "modularity_test_big.png")
    plot_partitioning(G, s_spec, "spectral_test_big.png")
    plot_partitioning(G, s_rand, "random_test_big.png")


if __name__ == "__main__":
    test_compare_small()
    test_compare_big()
