import networkx as nx
import time

from ising.flow import TOP, LOGGER
from ising.under_dev import MaxCutParser
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.benchmarks.parsers.Knapsack import QKP_parser
from ising.generators.TSP import TSP
from ising.generators.Knapsack import knapsack

from ising.under_dev.Partitioning.modularity import partitioning_modularity
from ising.under_dev.Partitioning.spectral_partitioning import spectral_partitioning

from ising.under_dev.Partitioning.plotter import plot_partitioning, plot_eigenvector, plot_percentage_cut_edges

from ising.utils.numpy import triu_to_symm



# def make_bar_chart_replica_nodes():
#     nb_cores = 2
#     profit, weights, capacity, _ = QKP_parser(TOP / "ising/benchmarks/Knapsack/jeu_100_25_1.txt")
#     model_knapsack = knapsack(profit, capacity, weights, 1.2)
#     s_mod_knapsack = partitioning_modularity(model_knapsack, nb_cores)
#     G_knapsack = nx.Graph(-2*triu_to_symm(model_knapsack.J))
#     replica_nodes_knapsack = plot_partitioning(G_knapsack, s_mod_knapsack, f"modularity_qkp_{nb_cores}.png")
#     orig_nodes_knapsack = np.unique(s_mod_knapsack, return_counts=True)[1]
#     perc_replica_QKP = np.mean([len(replica_nodes_knapsack[part]) / (orig_nodes_knapsack[idx]+len(replica_nodes_knapsack[part])) for idx, part in enumerate(np.unique(s_mod_knapsack))])*100

#     G1, _ = G_parser(TOP / "ising/benchmarks/G/G1.txt")
#     model_MC = MaxCutParser.generate_maxcut(G1)
#     s_mod_MC = partitioning_modularity(model_MC, nb_cores)
#     G_MC = nx.Graph(-2*triu_to_symm(model_MC.J))
#     replica_nodes_MC = plot_partitioning(G_MC, s_mod_MC, f"modularity_MC_{nb_cores}.png")
#     orig_nodes_MC = np.unique(s_mod_MC, return_counts=True)[1]
#     perc_replica_MC = np.mean([len(replica_nodes_MC[part]) / (orig_nodes_MC[idx]+len(replica_nodes_MC[part])) for idx, part in enumerate(np.unique(s_mod_MC))])*100

#     burma14, _ = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
#     model_TSP = TSP(burma14, 1.2)
#     s_mod_TSP = partitioning_modularity(model_TSP, nb_cores)
#     replica_nodes_TSP = plot_partitioning(burma14, s_mod_TSP, f"modularity_TSP_{nb_cores}.png")
#     orig_nodes_TSP = np.unique(s_mod_TSP, return_counts=True)[1]
#     perc_replica_TSP = np.mean([len(replica_nodes_TSP[part]) / (orig_nodes_TSP[idx]+len(replica_nodes_TSP[part])) for idx, part in enumerate(np.unique(s_mod_TSP))])*100

#     plt.figure()
#     plt.bar(["Knapsack", "Max Cut", "TSP"], [perc_replica_QKP, perc_replica_MC, perc_replica_TSP])
#     plt.ylabel("Percentage of replica nodes in partition")
#     plt.xlabel("Problem")
#     plt.savefig(figtop / "percentage_replica_nodes_partitioning.png")
#     plt.close()

def compare_techniques():
    LOGGER.info("========== Max Cut ==========")
    G16, _ = MaxCutParser.G_parser(TOP / "ising/benchmarks/G/G16.txt")
    model = MaxCutParser.generate_maxcut(G16)
    nb_cores = 2

    s_time = time.time()
    spectral_s, spectral_v, mean = spectral_partitioning(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for normalized cut: %s", e_time-s_time)
    plot_eigenvector(spectral_v, mean, "MCP_eigenvector_normalized_cut.png")
    plot_partitioning(G16, spectral_s, "MCP_partitioning_normalized_cut.png")

    s_time = time.time()
    mod_s, (mod_v, mean) = partitioning_modularity(model, 2)
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
    mod_s, (mod_v, mean) = partitioning_modularity(model, 2)
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
    mod_s, (mod_v, mean) = partitioning_modularity(model, 2)
    e_time = time.time()
    LOGGER.info("total time needed for modularity cut: %s", e_time-s_time)
    plot_eigenvector(mod_v, mean, "TSP_eigenvector_modularity_cut.png")
    plot_partitioning(GTSP, mod_s, "TSP_partitioning_modularity_cut.png")
    plot_percentage_cut_edges(model, {"normalized cut":spectral_s, "modularity cut":mod_s}, "TSP_cut_Edges.png")

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
    model_MC = MaxCut(G1)
    s_mod_MC = partitioning_modularity(model_MC, nb_cores)
    G_MC = nx.Graph(-2*triu_to_symm(model_MC.J))
    replica_nodes_MC = plot_partitioning(G_MC, s_mod_MC, f"modularity_MC_{nb_cores}.png")
    orig_nodes_MC = np.unique(s_mod_MC, return_counts=True)[1]
    perc_replica_MC = np.mean([len(replica_nodes_MC[part]) / (orig_nodes_MC[idx]+len(replica_nodes_MC[part])) for idx, part in enumerate(np.unique(s_mod_MC))])*100

    burma14, _ = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model_TSP = TSP(burma14, 1.2)
    s_mod_TSP = partitioning_modularity(model_TSP, nb_cores)
    G_TSP = nx.Graph(-2*triu_to_symm(model_TSP.J))
    replica_nodes_TSP = plot_partitioning(G_TSP, s_mod_TSP, f"modularity_TSP_{nb_cores}.png")
    orig_nodes_TSP = np.unique(s_mod_TSP, return_counts=True)[1]
    perc_replica_TSP = np.mean([len(replica_nodes_TSP[part]) / (orig_nodes_TSP[idx]+len(replica_nodes_TSP[part])) for idx, part in enumerate(np.unique(s_mod_TSP))])*100

    plt.figure()
    plt.bar(["Knapsack", "Max Cut", "TSP"], [perc_replica_QKP, perc_replica_MC, perc_replica_TSP])
    plt.ylabel("Percentage of replica nodes in partition")
    plt.xlabel("Problem")
    plt.savefig(figtop / "percentage_replica_nodes_partitioning.png")
    plt.close()


if __name__ == "__main__":
    # make_bar_chart_replica_nodes()
    compare_techniques()
