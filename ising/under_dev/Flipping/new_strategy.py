import numpy as np
import matplotlib.pyplot as plt

from ising.solvers.Multiplicative import Multiplicative

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP, get_TSP_value
from ising.postprocessing.summarize_energies import summary_energies
from ising.utils.helper_functions import return_rx

np.random.seed(1)
figtop = TOP / "ising/under_Dev/Flipping"

def find_cluster_Wolff(model, sigma, start_node, P_add, max_size):
    stack = [start_node]
    cluster = [start_node]

    coupling = model.J + model.J.T

    spins = np.arange(model.num_variables)
    for node in stack:
        neighbours = spins[(coupling[node, :] != 0.)]

        for neighbour in neighbours:
            if neighbour in cluster:
                continue
            
            rand_val = np.random.uniform(0, 1)
            if sigma[neighbour] == sigma[start_node] and rand_val < P_add:
                cluster.append(neighbour)
                stack.append(neighbour)
        
        stack.remove(node)
        if len(cluster) >= max_size:
            break
    return cluster

def find_cluster_mean(best_sigmas, sigma, max_size, threshold, option):
    mean_sigma = np.mean(best_sigmas, axis=0)

    if option == 1:
        available_nodes = np.where(np.abs(mean_sigma) >= threshold, np.arange(len(sigma)), -1)
    else:
        available_nodes = np.where(np.abs(mean_sigma) <= threshold, np.arange(len(sigma)), -1)
        if len(available_nodes[available_nodes >= 0]) < max_size:
            current_size = len(available_nodes[available_nodes >= 0])
            ind_unavailable_nodes = np.where(available_nodes < 0)[0]
            chosen_nodes = np.random.choice(ind_unavailable_nodes, (max_size - current_size,))
            available_nodes[chosen_nodes] = np.arange(len(available_nodes))[chosen_nodes]
    
    cluster = np.random.choice(available_nodes[available_nodes >= 0], size=(max_size,))
    return cluster

def do_flipping(model, sigma_init:np.ndarray, cluster_size_init:int, cluster_size_end:int, nb_flipping:int, dt:float, num_iterations:int, cluster_threshold:float):
    sigma = sigma_init.copy()
    energies = []

    best_sigmas = []
    size_func = lambda x: int((return_rx(nb_flipping, cluster_size_init, cluster_size_end)**x) * cluster_size_init )

    prev_energy = np.inf
    prev_sigma = sigma.copy()
    for i in range(nb_flipping):
        sigma, energy, new_energies =  Multiplicative().solve(model, sigma, dt, num_iterations,
                                            initial_temp_cont=0.0)
        energies += new_energies
        if energy < prev_energy:
            LOGGER.info("Flip accepted")
            prev_energy = energy
            prev_sigma = sigma.copy()
            best_sigmas.append(sigma.copy())
            if len(best_sigmas) > 10:
                best_sigmas.pop(0)
        
        LOGGER.info(f"Energy: {energy:.2f} and best found energy: {prev_energy:.2f}")
        cluster_size = size_func(i)
        cluster = find_cluster_mean(best_sigmas, sigma, cluster_size, cluster_threshold, 1)

        LOGGER.info(f"Cluster size: {len(cluster)}")
        sigma = prev_sigma.copy()
        sigma[cluster] = -1*sigma[cluster]
    return energies, prev_energy, prev_sigma

def plot_data(data, figname, xlabel:str, xticks:list[str], ylabel:str, yticks:list[str], best_found:float, colorbar_label:str= "Energy"):
    plt.figure()
    ax = plt.gca()
    plt.imshow(data, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_xticks(range(len(xticks)), xticks, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_yticks(range(len(yticks)), yticks)
    plt.colorbar(label=colorbar_label, ticks=np.arange(best_found, np.max(data)+1, 200, dtype=int))
    plt.savefig(figname, bbox_inches="tight")
    plt.close()


def TSP_flipping():
    burma14_graph, best_found = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model_burma14 = TSP(burma14_graph, 1.2)
    LOGGER.info(f"Best found: {best_found}")

    dt = 1e-6
    num_iterations = 50000

    sigma = np.random.uniform(-1, 1, (model_burma14.num_variables,))
    cluster_sizes_end = [1/18, 1/16, 1/14, 1/12, 1/10, 1/8, 1/6, 1/4]
    str_init_sizes = ["1/18", "1/16", "1/14", "1/12", "1/10", "1/8", "1/6", "1/4"]
    cluster_sizes_init = [1/2, 2/3, 3/4, 4/5, 5/6, 7/8, 9/10, 1]
    str_end_sizes = ["1/2", "2/3", "3/4", "4/5", "5/6", "7/8", "9/10", "1"]
    flipping_length = 100 # 130
    mean_cluster_th = 0.7

    best_energies = np.zeros((len(cluster_sizes_init), len(cluster_sizes_end)))
    distance_energy = np.zeros((len(cluster_sizes_init), len(cluster_sizes_end)))

    for i,size_in in enumerate(cluster_sizes_init):
        init_size = int(model_burma14.num_variables * size_in)
        
        for j, size_en in enumerate(cluster_sizes_end):
            end_size = int(model_burma14.num_variables * size_en)
            energies, prev_energy, prev_sigma = do_flipping(model_burma14, sigma, cluster_size_init=init_size, cluster_size_end=end_size, nb_flipping=flipping_length, 
                                                            dt=dt, num_iterations=num_iterations, cluster_threshold=mean_cluster_th)
            best_energies[i,j] = prev_energy
            distance_energy[i,j] = get_TSP_value(burma14_graph, prev_sigma)
            LOGGER.info(f"Initial cluster size: {init_size}, final cluster size: {end_size}, energy: {prev_energy:.2f}, distance: {distance_energy[i,j]}, sigma: {prev_sigma}")

    plot_data(best_energies, figtop / "burma14_linear_change.png", "Final cluster size", str_end_sizes, "Initial cluster size", str_init_sizes, best_found)
    plot_data(distance_energy, figtop / "burma14_linear_change_distance.png", "Final cluster size", str_end_sizes, "Initial cluster size", str_init_sizes, best_found, "TSP distance")

def MaxCut_flipping():
    g1, best_found = G_parser(TOP / "ising/benchmarks/G/G1.txt")
    model_g1 = MaxCut(g1)
    sigma_init = np.random.choice([-1, 1], size=(model_g1.num_variables,))

    # Cluster parameters
    cluster_sizes = [1/15, 1/5, 1/4, 2/5, 1/2, 4/6, 9/10]
    str_sizes = ["1/15", "1/5", "1/4", "2/5", "1/2", "4/6", "9/10"]
    nb_flipping = [20, 50, 80, 100, 120, 150, 180]
    str_nb = ["20", "50", "80", "100", "120", "150", "180"]
    threshold = 0.7

    # Solver parameters
    dt = 1e-4
    num_iterations = 100000
    best_energies = np.zeros((len(cluster_sizes), len(nb_flipping)))

    for i, size in enumerate(cluster_sizes):
        cl_size = int(model_g1.num_variables * size)

        for j, nb in enumerate(nb_flipping):
            _, best_energy, prev_sigma = do_flipping(model_g1, sigma_init, cl_size, cl_size, nb, dt, num_iterations, threshold)
            best_energies[i, j] = best_energy
            LOGGER.info(f"Cluster size: {size}, amount of flips: {nb}, best energy: {best_energy:.2f}, prev_sigma: {prev_sigma}")

    plot_data(best_energies, figtop / "G1_flipping.png", "cluster size", str_sizes, "amount of flips", str_nb, best_found)

if __name__ == "__main__":
    TSP_flipping()
    # MaxCut_flipping()