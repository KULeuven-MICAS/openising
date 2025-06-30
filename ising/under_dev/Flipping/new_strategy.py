import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import seaborn as sns #type: ignore
import pandas as pd # type: ignore

from ising.solvers.Multiplicative import Multiplicative

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP, get_TSP_value
from ising.postprocessing.summarize_energies import summary_energies
from ising.utils.helper_functions import return_rx

np.random.seed(1)
figtop = TOP / "ising/under_dev/Flipping/Figures"

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

def find_cluster_gradient(model, sigma, max_size, threshold):
    coupling = model.J + model.J.T
    gradient = coupling @ sigma + model.h
    gradient /= np.max(gradient)
    available_nodes = np.where(np.abs(gradient) >= threshold, np.arange(len(sigma)), -1)
    if len(available_nodes[available_nodes >= 0]) < max_size:
        current_size = len(available_nodes[available_nodes >= 0])
        ind_unavailable_nodes = np.where(available_nodes < 0)[0]
        chosen_nodes = np.random.choice(ind_unavailable_nodes, (max_size - current_size,))
        available_nodes[chosen_nodes] = np.arange(len(available_nodes))[chosen_nodes]
    cluster = np.random.choice(available_nodes[available_nodes >= 0], size=(max_size,))
    return cluster


def do_flipping(cluster_size_init:int, cluster_size_end:int, file, cluster_threshold:float, sigma_init:np.ndarray, model, nb_flipping:int, dt:float, num_iterations:int):
    sigma = sigma_init.copy()
    energies = []

    # best_sigmas = []
    size_func = lambda x: int((return_rx(nb_flipping, cluster_size_init, cluster_size_end)**(x*3)) * (cluster_size_init-cluster_size_end) + cluster_size_end)

    prev_energy = np.inf
    prev_sigma = sigma.copy()
    for i in range(nb_flipping):
        sigma, energy =  Multiplicative().solve(model, sigma, dt, num_iterations,
                                            initial_temp_cont=0.0)
        energies.append(energy)
        if energy < prev_energy:
            prev_energy = energy
            prev_sigma = sigma.copy()
            # best_sigmas.append(sigma.copy())
            # if len(best_sigmas) > 10:
            #     best_sigmas.pop(0)
        
        cluster_size = size_func(i)
        LOGGER.info(f"Cluster size: {cluster_size}")
        cluster = find_cluster_gradient(model, prev_sigma, cluster_size, cluster_threshold, 1)
        with open(file, "a") as f:
            f.write(f"iteration {i}: energy {energy:.2f}")
        sigma = prev_sigma.copy()
        sigma[cluster] = -1*sigma[cluster]
    with open(file, "a") as f:
        f.write(f"initial size {cluster_size_init}, final size {cluster_size_end}, final energy {energy:.2f}")
    return energies, prev_energy, prev_sigma

def plot_data(data, figname, xlabel:str, xticks:list[str], ylabel:str, yticks:list[str], best_found:float, colorbar_label:str= "Energy"):
    plt.figure()
    ax = plt.gca()
    plt.imshow(data, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_xticks(range(len(xticks)), xticks, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_yticks(range(len(yticks)), yticks)
    plt.colorbar(label=colorbar_label, ticks=np.arange(best_found, np.max(data)+1, 100, dtype=int)) # 
    plt.savefig(figname, bbox_inches="tight")
    plt.close()

def make_bar_plot(dataframes:dict[str:pd.DataFrame], xaxis_name, yaxis_name, figname, best_found:float):
    # concat the dataframes
    plt.figure()
    if len(dataframes.keys()) > 1:
        data = dict()
        for df_name in dataframes.keys():
            data[df_name] = dataframes[df_name].melt()
        df = pd.concat(data, names=["source", "old_index"])
        df = df.reset_index(level=0).reset_index(drop=True)
        sns.boxplot(data=df, x='variable', y='value', hue="source")
    else:
        df = dataframes[list(dataframes.keys())[0]]
        sns.boxplot(data=df, x=xaxis_name, y=yaxis_name)
    plt.axhline(y=best_found, color='k', linestyle='--', label='Best found')
    plt.legend()
    plt.xlabel(xaxis_name)
    plt.ylabel(yaxis_name)
    plt.savefig(figtop / figname, bbox_inches="tight")
    plt.close()

def TSP_flipping():
    burma14_graph, best_found = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model_burma14 = TSP(burma14_graph, 1.2)
    LOGGER.info(f"Best found: {best_found}")

    dt = 1e-6
    num_iterations = 50000
    nb_runs = 20

    sigma = np.random.uniform(-1, 1, (model_burma14.num_variables,nb_runs))
    final_cluster_size = int(model_burma14.num_variables*1/18)
    init_cluster_size = int(model_burma14.num_variables*2/3)
    flipping_length = 100 # 130
    mean_cluster_th = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    tasks = [(mean_cluster_th[i], sigma[:,j]) for i in range(len(mean_cluster_th)) for j in range(nb_runs)]
    with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()/4)) as pool:
        flipping_partial = partial(do_flipping, model=model_burma14, 
                                                cluster_size_init=init_cluster_size, 
                                                cluster_size_end=final_cluster_size, 
                                                nb_flipping=flipping_length, 
                                                dt=dt, 
                                                num_iterations=num_iterations)
        results = pool.starmap(flipping_partial, tasks)

    all_energies = np.array([result[0] for result in results]).reshape((len(mean_cluster_th), nb_runs, flipping_length))

    for i in range(len(mean_cluster_th)):
        mean_energies = np.mean(all_energies[i, :, :], axis=0)
        max_energies = np.max(all_energies[i, :, :], axis=0)
        min_energies = np.min(all_energies[i, :, :], axis=0)
        std_energies = np.std(all_energies[i, :, :], axis=0)
        file = TOP / f"ising/under_dev/Flipping/threshold_{mean_cluster_th[i]}_burma14.csv"
        header = "mean min max std"
        np.savetxt(file, np.array([mean_energies, min_energies, max_energies, std_energies]), fmt="%.2f", header=header)

    best_energies = np.array([result[1] for result in results]).reshape((len(mean_cluster_th), nb_runs))
    best_energies = np.mean(best_energies, axis=1)
    plot_data(best_energies, figtop / "burma14_cluster_threshold.png", "Threshold", np.array(mean_cluster_th, dtype=str), "Final cluster size", ["1/18"], best_found)

    dfs = {f"threshold {mean_cluster_th[i]}": None for i in range(len(mean_cluster_th))}
    for i, threshold in enumerate(mean_cluster_th):
        dfs[f"threshold {threshold}"] = pd.DataFrame({iteration: all_energies[i, :, iteration] for iteration in range(flipping_length)})
    make_bar_plot(dfs, "Iteration", "Energy", "burma14_threshold_barplot.png", best_found)

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