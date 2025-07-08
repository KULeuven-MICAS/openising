import numpy as np
import multiprocessing
from functools import partial
import pandas as pd 
import os

from ising.solvers.Multiplicative import Multiplicative

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP
from ising.stages.model.ising import IsingModel
from ising.utils.flow import return_rx
from ising.under_dev.Flipping.Wolff_cluster import find_cluster_Wolff
from ising.under_dev.Flipping.gradient_cluster import find_cluster_gradient
from ising.under_dev.Flipping.mean_cluster import find_cluster_mean
from ising.under_dev.Flipping.frozen_gradient_cluster import find_frozen_gradient_cluster
from ising.under_dev.Flipping.plotting import plot_data, make_bar_plot
np.random.seed(1)
NICENESS = 1

def do_flipping(cluster_size_init:int, cluster_size_end:int, sigma_init:np.ndarray,cluster_threshold:float, model:IsingModel, nb_flipping:int, dt:float, num_iterations:int, cluster_choice:str=""):
    sigma = sigma_init.copy()
    energies = []
    nb_seeds = 2
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
        
        cluster_size = size_func(i)
        if cluster_choice=="random":
            cluster = np.random.choice(model.num_variables, size=(cluster_size,), replace=False)
        else:
            cluster = find_frozen_gradient_cluster(model, prev_sigma, cluster_threshold, nb_seeds, cluster_size)
        sigma = prev_sigma.copy()
        sigma[cluster] = -1*sigma[cluster]
    LOGGER.info(f"Threshold: {cluster_threshold}, Best energy: {prev_energy}")
    return energies, prev_energy, prev_sigma



def TSP_flipping():
    burma14_graph, best_found = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model_burma14 = TSP(burma14_graph, 1.2)
    LOGGER.info(f"Best found: {best_found}")

    dt = 1e-6
    num_iterations = 50000
    nb_runs = 20

    sigma = np.random.uniform(-1, 1, (model_burma14.num_variables,nb_runs))
    final_cluster_size = int(model_burma14.num_variables*1/14)
    init_cluster_size = int(model_burma14.num_variables*7/8)
    flipping_length = 130
    mean_cluster_th = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    str_thresholds = ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    # do_flipping(cluster_size_init=init_cluster_size, cluster_size_end=final_cluster_size, sigma_init=sigma[:,0], cluster_threshold=mean_cluster_th, model=model_burma14, nb_flipping=flipping_length, dt=dt, num_iterations=num_iterations)
    tasks = [(init_cluster_size, final_cluster_size, sigma[:,j], mean_cluster_th[i]) for i in range(len(mean_cluster_th)) for j in range(nb_runs)]
    with multiprocessing.Pool(initializer=os.nice, initargs=(NICENESS,)) as pool:
        flipping_partial = partial(do_flipping, model=model_burma14, 
                                                nb_flipping=flipping_length, 
                                                dt=dt, 
                                                num_iterations=num_iterations)
        results = pool.starmap(flipping_partial, tasks)

    all_energies = np.array([result[0] for result in results]).reshape((len(mean_cluster_th), nb_runs, flipping_length))

    # for i in range(len(mean_cluster_th)):
    #     mean_energies = np.mean(all_energies[i, :, :], axis=0)
    #     max_energies = np.max(all_energies[i, :, :], axis=0)
    #     min_energies = np.min(all_energies[i, :, :], axis=0)
    #     std_energies = np.std(all_energies[i, :, :], axis=0)
    #     file = TOP / f"ising/under_dev/Flipping/threshold_{mean_cluster_th[i]}_burma14.csv"
    #     header = "mean min max std"
    #     np.savetxt(file, np.array([mean_energies, min_energies, max_energies, std_energies]).T, fmt="%.2f", header=header)

    best_energies = np.array([result[1] for result in results]).reshape((len(mean_cluster_th), nb_runs))
    best_energies = np.mean(best_energies, axis=1).reshape((1, -1))
    plot_data(best_energies, "burma14_cluster_threshold.png", "Threshold", str_thresholds, "Final cluster size", ["1/18"], best_found)

    # dfs = {f"threshold {mean_cluster_th[i]}": None for i in range(len(mean_cluster_th))}
    # for i, threshold in enumerate(mean_cluster_th):
    #     dfs[f"threshold {threshold}"] = pd.DataFrame({iteration: all_energies[i, :, iteration] for iteration in range(flipping_length)})
    # make_bar_plot(dfs, "Iteration", "Energy", "burma14_threshold_barplot.png", best_found)

def TSP_flipping_more_annealing():
    burma14_graph, best_found = TSP_parser(TOP / "ising/benchmarks/TSP/burma14.tsp")
    model_burma14 = TSP(burma14_graph, 1.2)
    LOGGER.info(f"Best found: {best_found}")

    dt = 1e-6
    num_iterations = 50000
    nb_runs = 20

    sigma = np.random.uniform(-1, 1, (model_burma14.num_variables,nb_runs))
    final_cluster_size = int(model_burma14.num_variables*1/18)
    init_cluster_size = int(model_burma14.num_variables*2/3)
    flipping_lengths = [10, 130] 
    threshold = 1.0

    tasks = [(flipping_length, sigma[:, j]) for flipping_length in flipping_lengths for j in range(nb_runs)]
    with multiprocessing.Pool(initializer=os.nice, initargs=(NICENESS,)) as pool:
        flipping_partial = partial(do_multiple_annealing, model=model_burma14, 
                                                cluster_size_init=init_cluster_size, 
                                                cluster_size_end=final_cluster_size,
                                                threshold=threshold, 
                                                dt=dt, 
                                                num_iterations=num_iterations)
        results = pool.starmap(flipping_partial, tasks)
    
    dfs = {f"#flipping {flipping_lengths[i]}": None for i in range(len(flipping_lengths))}
    all_energies = np.array([result[0] for result in results]).reshape((len(flipping_lengths), nb_runs, 100))
    for i, length in enumerate(flipping_lengths):
        dfs[f"#flipping {length}"] = pd.DataFrame({iteration: all_energies[i, :, iteration] for iteration in range(100)})
    make_bar_plot(dfs, "Iteration", "Energy", "burma14_annealing_barplot.png", best_found)

    
def do_multiple_annealing(nb_flipping, sigma_init, model, cluster_size_init, cluster_size_end, dt, num_iterations, threshold):
    if nb_flipping == 10:
        energies = []
        prev_energies = []
        for i in range(13):
            energies_current, prev_energy, prev_sigma = do_flipping(cluster_size_init=cluster_size_init, 
                                                                    cluster_size_end=cluster_size_end, 
                                                                    cluster_threshold=threshold, 
                                                                    sigma_init=sigma_init, 
                                                                    model=model, 
                                                                    nb_flipping=nb_flipping, 
                                                                    dt=dt, 
                                                                    num_iterations=num_iterations)
            energies.append(energies_current)
            prev_energies.append(prev_energy)
            sigma_init = prev_sigma.copy()
        prev_energy = np.min(prev_energies)
        LOGGER.info(f"flipping 10: energies: {energies}")
    else:
        energies, prev_energy, prev_sigma = do_flipping(cluster_size_init=cluster_size_init, 
                                                        cluster_size_end=cluster_size_end, 
                                                        cluster_threshold=threshold, 
                                                        sigma_init=sigma_init, 
                                                        model=model, 
                                                        nb_flipping=nb_flipping, 
                                                        dt=dt, 
                                                        num_iterations=num_iterations)
        LOGGER.info(f"flipping 100: energies: {energies}")
    return energies, prev_energy, prev_sigma

if __name__ == "__main__":
    TSP_flipping()
    # TSP_flipping_more_annealing()
