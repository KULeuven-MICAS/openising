import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
import time

from ising.flow import TOP, LOGGER

from ising.generators.MaxCut import MaxCut, random_MaxCut
from ising.model.ising import IsingModel

from ising.solvers.exhaustive import ExhaustiveSolver
from ising.solvers.SB import ballisticSB

from ising.under_dev.Partitioning.modularity import partitioning_modularity
from ising.under_dev.Partitioning.spectral_partitioning import spectral_partitioning
from ising.under_dev.Partitioning.random_partitioning import random_partitioning

from ising.under_dev.Partitioning.apply_partitioning import apply_partitioning
from ising.under_dev.Partitioning.dual_decomposition import dual_decomposition

from ising.utils.flow import make_directory, return_c0
from ising.utils.numpy import triu_to_symm

figtop = TOP / "ising/under_dev/Partitioning/plots"
make_directory(figtop)
logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)


def plot_partitioning(G:nx.Graph, s, fig_name):
    pos = nx.spring_layout(G, k=5 / np.sqrt(5))
    _, (ax1, ax2) = plt.subplots(2, 1)

    nx.draw_networkx(G, pos, ax=ax1)
    unique_partitions = np.unique(s)
    colors = plt.cm.tab10(np.arange(len(unique_partitions)))

    for idx, partition in enumerate(unique_partitions):
        nodes_in_partition = [node for node in G.nodes if s[node] == partition]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_partition, node_color=[colors[idx]], ax=ax2)
    for edge in G.edges:
        if s[edge[0]] != s[edge[1]]:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax2, style="--")
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax2)
    nx.draw_networkx_labels(G, pos, ax=ax2)

    plt.savefig(figtop / fig_name)

def optimal_state_from_partitioning(optimal_states:dict[int: np.ndarray], model: IsingModel, partitioning: np.ndarray, replica_nodes: dict[int:np.ndarray]):
    state = np.zeros((model.num_variables,))
    partitions = np.unique(partitioning)
    
    nodes_partitions = {i:[] for i in np.unique(partitioning)}
    node_maps = dict()
    for node, part in enumerate(partitioning):
        nodes_partitions[part].append(node)

    for _, part in enumerate(partitions):
        part_nodes = set(nodes_partitions[part])
        part_nodes = list(part_nodes | replica_nodes[part])
        part_nodes.sort()

        node_map = {node: idx for idx, node in enumerate(part_nodes)}
        node_maps[part] = node_map


    for node, part in enumerate(partitioning):
        amount_replicas = 3
        avg_node = 0
        for other_part, replica_node in replica_nodes.items():
            if node in replica_node and other_part != part:
                amount_replicas += 1
                avg_node += optimal_states[other_part][node_maps[other_part][node]]
        avg_node += optimal_states[part][node_maps[part][node]]*3
        avg_node /= amount_replicas
        if avg_node == 0:
            state[node] = optimal_states[part][node_maps[part][node]]
        else:
            state[node] = np.sign(avg_node)

    energy = model.evaluate(state)

    return state, energy

def plot_energies_cores(cores:list[int], energies:list[float], best_energy:float,  figname:str):

    plt.figure()
    plt.plot(cores, energies, "x-")
    plt.axhline(best_energy, linestyle="--", color="k", label="Best energy")
    plt.xlabel("Number of cores")
    plt.ylabel("Optimal Energy")
    plt.legend()
    plt.savefig(figtop / figname)
    plt.close()

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

def test_accuracy():
    model = random_MaxCut(50, 8)
    # state, energy = ExhaustiveSolver().solve(model)
    state, best_energy = ballisticSB().solve(model, np.random.choice([-1, 1], size=(model.num_variables, )), 1000, 0.7/np.sqrt(model.num_variables), 0.25, 1.0)
    LOGGER.info(f"Optimal energy is {best_energy} and state is {state}")
    G = nx.Graph(-2*triu_to_symm(model.J))

    cores = range(2, 11)
    energies = []
    for nb_cores in cores:
        s_mod = partitioning_modularity(model, nb_cores)
        # LOGGER.info(f"Different partitions: {s_mod}")
        # plot_partitioning(G, s_mod, f"modularity_test_{nb_cores}.png")
        models, constraints, replica_nodes = apply_partitioning(model, s_mod)

        initial_states = {part: np.random.choice([-1, 1], size=(m.num_variables,), p=[0.5, 0.5]) for part, m in models.items()}
        hyperparameters = {"a0": 1, "c0": return_c0(model), "dtSB": 0.25}
        optimal_states, lambda_k = dual_decomposition(models, constraints, initial_states, 1000, "bSB", 1e-3, 0.0, **hyperparameters)
        # LOGGER.info(f"At optimal point, the lagrange parameters are {lambda_k}")

        state, energy = optimal_state_from_partitioning(optimal_states, model, s_mod, replica_nodes)
        energies.append(energy)
        LOGGER.info(f"Solution of dual decomposition is: {state} with energy: {energy}")

    plot_energies_cores(cores, energies, best_energy, "energies_partitioning.png")

if __name__ == "__main__":
    # test_compare_small()
    # test_compare_big()
    # test_dual_decomposition()
    test_accuracy()
