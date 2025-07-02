import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from argparse import Namespace

from ising.flow import TOP, LOGGER

from ising.benchmarks.parsers.G import G_parser
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.benchmarks.parsers.Knapsack import QKP_parser
from ising.stages.maxcut_parser_stage import MaxcutParserStage
from ising.stages.main_stage import MainStage
from ising.generators.TSP import TSP
from ising.generators.Knapsack import knapsack
from ising.stages.model.ising import IsingModel

from ising.solvers.exhaustive import ExhaustiveSolver

from ising.under_dev.Partitioning.modularity import partitioning_modularity
from ising.under_dev.Partitioning.spectral_partitioning import spectral_partitioning
from ising.under_dev.Partitioning.random_partitioning import random_partitioning

from ising.under_dev.Partitioning.apply_partitioning import apply_partitioning

from ising.utils.helper_functions import make_directory, return_c0
from ising.utils.numpy import triu_to_symm

figtop = TOP / "ising/under_dev/Partitioning/plots"
make_directory(figtop)
config = Namespace(benchmark="./ising/benchmarks/MaxCut/G22.txt",)
MaxCutParser = MaxcutParserStage(list_of_callables=[MainStage], config=config)

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
    plt.savefig(figtop / figname, bbox_inches='tight')
    plt.close()

def make_bar_chart_replica_nodes():
    nb_cores = 2
    profit, weights, capacity, _ = QKP_parser(TOP / "ising/benchmarks/Knapsack/jeu_100_25_1.txt")
    model_knapsack = knapsack(profit, capacity, weights, 1.2)
    s_mod_knapsack = partitioning_modularity(model_knapsack, nb_cores)
    G_knapsack = nx.Graph(-2*triu_to_symm(model_knapsack.J))
    replica_nodes_knapsack = plot_partitioning(G_knapsack, s_mod_knapsack, f"modularity_qkp_{nb_cores}.png")
    orig_nodes_knapsack = np.unique(s_mod_knapsack, return_counts=True)[1]
    perc_replica_QKP = np.mean([len(replica_nodes_knapsack[part]) / (orig_nodes_knapsack[idx]+len(replica_nodes_knapsack[part])) for idx, part in enumerate(np.unique(s_mod_knapsack))])*100

    G1, _ = G_parser(TOP / "ising/benchmarks/G/G1.txt")
    model_MC = MaxCutParser.generate_maxcut(G1)
    s_mod_MC = partitioning_modularity(model_MC, nb_cores)
    G_MC = nx.Graph(-2*triu_to_symm(model_MC.J))
    replica_nodes_MC = plot_partitioning(G_MC, s_mod_MC, f"modularity_MC_{nb_cores}.png")
    orig_nodes_MC = np.unique(s_mod_MC, return_counts=True)[1]
    perc_replica_MC = np.mean([len(replica_nodes_MC[part]) / (orig_nodes_MC[idx]+len(replica_nodes_MC[part])) for idx, part in enumerate(np.unique(s_mod_MC))])*100

    burma14, _ = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model_TSP = TSP(burma14, 1.2)
    s_mod_TSP = partitioning_modularity(model_TSP, nb_cores)
    replica_nodes_TSP = plot_partitioning(burma14, s_mod_TSP, f"modularity_TSP_{nb_cores}.png")
    orig_nodes_TSP = np.unique(s_mod_TSP, return_counts=True)[1]
    perc_replica_TSP = np.mean([len(replica_nodes_TSP[part]) / (orig_nodes_TSP[idx]+len(replica_nodes_TSP[part])) for idx, part in enumerate(np.unique(s_mod_TSP))])*100

    plt.figure()
    plt.bar(["Knapsack", "Max Cut", "TSP"], [perc_replica_QKP, perc_replica_MC, perc_replica_TSP])
    plt.ylabel("Percentage of replica nodes in partition")
    plt.xlabel("Problem")
    plt.savefig(figtop / "percentage_replica_nodes_partitioning.png")
    plt.close()

def compare_techniques():
    LOGGER.info("========== Max Cut ==========")
    G16, _ = G_parser(TOP / "ising/benchmarks/G/G16.txt")
    model = MaxCutParser.generate_maxcut(G16)
    nb_cores = 2

    s_time = time.time()
    spectral_s, spectral_v, mean = spectral_partitioning(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for normalized cut: %s", e_time-s_time)
    plot_eigenvector(spectral_v, mean, "MCP_eigenvector_normalized_cut.png")
    plot_partitioning(G16, spectral_s, "MCP_partitioning_normalized_cut.png")

    s_time = time.time()
    mod_s, mod_v, mean = partitioning_modularity(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for modularity cut: %s", e_time-s_time)
    plot_eigenvector(mod_v, mean, "MCP_eigenvector_modularity_cut.png")
    plot_partitioning(G16, mod_s, "MCP_partitioning_modularity_cut.png")

    plot_percentage_cut_edges(model, {"normalized cut":spectral_s, "modularity cut":mod_s}, "MCP_cut_Edges.png")
    LOGGER.info("========== Quadratic Knapsack ==========")

    profits, weights, capacity, _ = QKP_parser(TOP / "ising/benchmarks/Knapsack/jeu_100_25_1.txt")
    model = knapsack(profits, capacity, weights, 1.2)
    G_QKP = nx.Graph(-2*triu_to_symm(model.J))

    s_time = time.time()
    spectral_s, spectral_v, mean = spectral_partitioning(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for normalized cut: %s", e_time-s_time)
    plot_eigenvector(spectral_v, mean, "QKP_eigenvector_normalized_cut.png")
    plot_partitioning(G_QKP, spectral_s, "QKP_partitioning_normalized_cut.png")

    s_time = time.time()
    mod_s, mod_v, mean = partitioning_modularity(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for modularity cut: %s", e_time-s_time)
    plot_eigenvector(mod_v, mean, "QKP_eigenvector_modularity_cut.png")
    plot_partitioning(G_QKP, mod_s, "QKP_partitioning_modularity_cut.png")
    plot_percentage_cut_edges(model, {"normalized cut":spectral_s, "modularity cut":mod_s}, "QKP_cut_Edges.png")

    LOGGER.info("========== Traveling Salesman ==========")

    burma14, _ = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model = TSP(burma14, 1.2)
    GTSP = nx.Graph(-2*triu_to_symm(model.J))

    s_time = time.time()
    spectral_s, spectral_v, mean = spectral_partitioning(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for normalized cut: %s", e_time-s_time)
    plot_eigenvector(spectral_v, mean, "TSP_eigenvector_normalized_cut.png")
    plot_partitioning(GTSP, spectral_s, "TSP_partitioning_normalized_cut.png")

    s_time = time.time()
    mod_s, mod_v, mean = partitioning_modularity(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for modularity cut: %s", e_time-s_time)
    plot_eigenvector(mod_v, mean, "TSP_eigenvector_modularity_cut.png")
    plot_partitioning(GTSP, mod_s, "TSP_partitioning_modularity_cut.png")
    plot_percentage_cut_edges(model, {"normalized cut":spectral_s, "modularity cut":mod_s}, "TSP_cut_Edges.png")

if __name__ == "__main__":
    # make_bar_chart_replica_nodes()
    compare_techniques()
