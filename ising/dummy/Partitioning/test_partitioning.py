import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pathlib

from ising.generators.MaxCut import MaxCut
from ising.model.ising import IsingModel

from ising.solvers.exhaustive import ExhaustiveSolver

from ising.dummy.Partitioning.modularity import partitioning_modularity
from ising.dummy.Partitioning.spectral_partitioning import spectral_partitioning
from ising.dummy.Partitioning.random_partitioning import random_partitioning

from ising.dummy.Partitioning.dual_decomposition import dual_decomposition

from ising.utils.flow import make_directory
from ising.utils.numpy import triu_to_symm

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



def apply_partitioning(model, partitioning):
    nodes_1 = []
    nodes_2 = []

    # Separate the nodes based on the partitioning
    for node, part in enumerate(partitioning):
        if part == 1:
            nodes_1.append(node)
        else:
            nodes_2.append(node)

    # Replicate nodes in order to have fully separated models
    replica_nodes = set()
    triu_J = triu_to_symm(model.J)
    for node1 in nodes_1:
        connected_nodes = np.nonzero(triu_J[node1, :])[0]
        for other_node in connected_nodes:
            if other_node in nodes_2:
                replica_nodes.add(other_node)
                replica_nodes.add(node1)
    
    nodes_1 = list(set(nodes_1) | (replica_nodes))
    nodes_1.sort()
    nodes_2 = list(set(nodes_2) | (replica_nodes))
    nodes_2.sort()
    replica_nodes = list(replica_nodes)

    n1 = len(nodes_1)
    n2 = len(nodes_2)

    J1 = np.zeros((n1, n1))
    h1 = np.zeros((n1,))

    J2 = np.zeros((n2, n2))
    h2 = np.zeros((n2,))

    # Fill in J1 and h1
    for i, node_i in enumerate(nodes_1):
        h1[i] = model.h[node_i]
        for j, node_j in enumerate(nodes_1):
            J1[i, j] = model.J[node_i, node_j]

    # Fill in J2 and h2
    for i, node_i in enumerate(nodes_2):
        h2[i] = model.h[node_i]
        for j, node_j in enumerate(nodes_2):
            J2[i, j] = model.J[node_i, node_j]

    A = np.zeros((len(replica_nodes), n1))
    C = np.zeros((len(replica_nodes), n2))

    map1 = {node: idx for idx, node in enumerate(nodes_1)}
    map2 = {node: idx for idx, node in enumerate(nodes_2)}

    # For each replica node, add agreement constraints
    for rep_ind, node in enumerate(replica_nodes):
        idx1 = map1[node]  # Index in first partition
        idx2 = map2[node]  # Index in second partition
        
        # Add diagonal entries for the agreement constraints
        A[rep_ind, idx1] = 1
        C[rep_ind, idx2] = -1
    
    return IsingModel(J1, h1), IsingModel(J2, h2), A, C, replica_nodes

def optimal_state_from_partitioning(s1, s2, model: IsingModel, partitioning, replica_nodes):
    state = np.zeros((model.num_variables,))
    nodes_1 = []
    nodes_2 = []

    for node, part in enumerate(partitioning):
        if part == 1:
            nodes_1.append(node)
        else:
            nodes_2.append(node)
    
    nodes_1 = list(set(nodes_1) | set(replica_nodes))
    nodes_2 = list(set(nodes_2) | set(replica_nodes))

    map1 = {node:ind for ind, node in enumerate(nodes_1)}
    map2 = {node:ind for ind, node in enumerate(nodes_2)}

    for node, part in enumerate(partitioning):
        if node in replica_nodes and s1[map1[node]] != s2[map2[node]]:
            state[node] = s1[map1[node]] if part == 1 else s2[map2[node]]
        elif part == 1:
            state[node] = s1[map1[node]]
        else:
            state[node] = s2[map2[node]]
    
    energy = model.evaluate(state)

    return state, energy
            
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


def test_dual_decomposition():
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
    plot_partitioning(G, s_mod, "modularity_test_dual.png")
    plot_partitioning(G, s_spec, "spectral_test_dual.png")
    # plot_partitioning(G, s_mod, "modularity_test_dual.png")
    hyperparameters = {"initial_temp": 50, "cooling_rate": 0.9, "seed": 1}

    print("Get solution with exhaustive solver")
    state, energy = ExhaustiveSolver().solve(model)
    print(f"Optimal energy is {energy} and state is {state}")

    print(hyperparameters)
    print("--------------------")
    print("Solving with modularity partitioning")
    model1, model2, A, C, replica_nodes = apply_partitioning(model, s_mod)
    s1 = np.random.choice([-1, 1], size=(model1.num_variables,), p=[0.5, 0.5])
    s2 = np.random.choice([-1, 1], size=(model2.num_variables,), p=[0.5, 0.5])
    s1, s2, lambda_k = dual_decomposition((s1, model1), (s2, model2), A, C, 100, "SA", 0.01, **hyperparameters)

    print(f"At optimal point, the lagrange parameters are {lambda_k}")
    state, energy = optimal_state_from_partitioning(s1, s2, model, s_mod, replica_nodes)

    print(f"Solution of dual decomposition is: {state} with energy: {energy}")
    plot_partitioning(G, state, "optimal_solution_modularity.png")
    print("--------------------")
    print("Solving with specular partitioning")
    print(s_spec)
    model1, model2, A, C, replica_nodes = apply_partitioning(model, s_spec)
    s1 = np.random.choice([-1, 1], size=(model1.num_variables,), p=[0.5, 0.5])
    s2 = np.random.choice([-1, 1], size=(model2.num_variables,), p=[0.5, 0.5])

    s1, s2, lambda_k = dual_decomposition((s1, model1), (s2, model2), A, C, 200, "SA", 0.01, **hyperparameters)

    print(f"At optimal point, the lagrange parameters are {lambda_k}")
    state, energy = optimal_state_from_partitioning(s1, s2, model, s_spec, replica_nodes)
    plot_partitioning(G, state, "optimal_solution_spectral.png")


    print(f"Solution of dual decomposition is: {state} with energy: {energy}")


if __name__ == "__main__":
    # test_compare_small()
    # test_compare_big()
    test_dual_decomposition()
