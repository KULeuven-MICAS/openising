import numpy as np
import multiprocessing
from functools import partial
import pandas as pd 
import os

from ising.solvers.Multiplicative import Multiplicative

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP
from ising.under_dev import MaxCutParser
# from ising.benchmarks.parsers.Knapsack import QKP_parser
# from ising.generators.Knapsack import knapsack
from ising.stages.model.ising import IsingModel
from ising.utils.flow import return_rx
from ising.under_dev.Flipping.Wolff_cluster import find_cluster_Wolff
from ising.under_dev.Flipping.gradient_cluster import find_cluster_gradient, find_cluster_gradient_largest
from ising.under_dev.Flipping.mean_cluster import find_cluster_mean
from ising.under_dev.Flipping.frozen_gradient_cluster import find_frozen_gradient_cluster, approximate_frozen_gradient_cluster
from ising.under_dev.Flipping.plotting import plot_data, make_bar_plot
np.random.seed(1)
NICENESS = 0

def do_flipping(cluster_size_init:int, cluster_size_end:int, sigma_init:np.ndarray,cluster_threshold:float, model:IsingModel, nb_flipping:int, dt:float, num_iterations:int, cluster_choice:str=""):
    sigma = sigma_init.copy()
    energies = []
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
            # LOGGER.info(f"Current best energy: {prev_energy}")
        
        cluster_size = size_func(i)
        # LOGGER.info(f"Amount of seeds: {int(cluster_size / cluster_size_end)}")
        if cluster_choice=="random":
            cluster = np.random.choice(model.num_variables, size=(cluster_size,), replace=False)
        else:
            cluster = find_cluster_gradient(model, prev_sigma, cluster_size, cluster_threshold)
        sigma = prev_sigma.copy()
        sigma[cluster] *= -1
    LOGGER.info(f"Init cluster size: {cluster_size_init}, final cluster size: {cluster_size_end}, Best energy: {prev_energy}")
    return energies, prev_energy, prev_sigma


def TSP_flipping():
    graph, best_found = MaxCutParser.G_parser(TOP / "ising/benchmarks/G/K2000.txt")
    model_burma14 = MaxCutParser.generate_maxcut(graph)
    LOGGER.info(f"Best found: {best_found}")

    dt = 1e-4
    num_iterations = 50000
    nb_runs = 10

    sigma = np.random.uniform(-1, 1, (model_burma14.num_variables,nb_runs))
    final_cluster_size = int(model_burma14.num_variables*1/50)
    init_cluster_size = int(model_burma14.num_variables)
    flipping_length = 130
    # threshold = 0.8
    mean_cluster_th = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    str_cluster = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    # do_flipping(cluster_size_init=init_cluster_size, cluster_size_end=final_cluster_size, sigma_init=sigma[:,0], cluster_threshold=mean_cluster_th, model=model_burma14, nb_flipping=flipping_length, dt=dt, num_iterations=num_iterations)
    tasks = [(init_cluster_size, final_cluster_size, sigma[:,j], threshold) for j in range(nb_runs) for threshold in mean_cluster_th]
    with multiprocessing.Pool(initializer=os.nice, initargs=(NICENESS,)) as pool:
        flipping_partial = partial(do_flipping, model=model_burma14, 
                                                nb_flipping=flipping_length, 
                                                dt=dt, 
                                                num_iterations=num_iterations)
        results = pool.starmap(flipping_partial, tasks)


    best_energies = np.array([result[1] for result in results]).reshape((len(mean_cluster_th),nb_runs))
    best_energies = np.mean(best_energies, axis=1).reshape((len(mean_cluster_th),1))
    plot_data(best_energies, "MCP_cluster_size.png", "Threshold", str_cluster, "Start cluster size", "1", best_found)


if __name__ == "__main__":
    TSP_flipping()
