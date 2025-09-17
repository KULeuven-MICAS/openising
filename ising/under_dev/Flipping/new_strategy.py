import numpy as np
import multiprocessing
from functools import partial
import os

from ising.solvers.Multiplicative import Multiplicative

from ising.flow import TOP, LOGGER
from ising.under_dev import TSPParser, QKPParser, MaxCutParser
from ising.generators.TSP import TSP
# from ising.under_dev import MaxCutParser
from ising.stages.model.ising import IsingModel
from ising.utils.flow import return_rx
from ising.under_dev.Flipping.Wolff_cluster import find_cluster_Wolff
from ising.under_dev.Flipping.gradient_cluster import find_cluster_gradient, find_cluster_gradient_largest
from ising.under_dev.Flipping.mean_cluster import find_cluster_mean
from ising.under_dev.Flipping.frozen_gradient_cluster import find_frozen_gradient_cluster, approximate_frozen_gradient_cluster
from ising.under_dev.Flipping.smoothening_cluster import smoothening_cluster
from ising.under_dev.Flipping.frequency_cluster import frequency_cluster
from ising.under_dev.Flipping.plotting import plot_data, make_bar_plot
np.random.seed(1)
NICENESS = 0
AMOUNT_CORES = os.getenv("AMOUNT_CORES")

def do_flipping(cluster_size_init:int, cluster_size_end:int, size_change:int, sigma_init:np.ndarray,cluster_threshold:float, model:IsingModel, nb_flipping:int, dt:float, num_iterations:int, cluster_choice:str=""):
    sigma = sigma_init.copy()
    energies = []
    size_func = lambda x: int((return_rx(nb_flipping, cluster_size_init, cluster_size_end)**(x*size_change)) * (cluster_size_init-cluster_size_end) + cluster_size_end)

    prev_energy = np.inf
    prev_sigma = sigma.copy()
    best_sigmas = [prev_sigma.copy()]
    points = []
    for i in range(nb_flipping):
        sigma, energy =  Multiplicative().solve(model, sigma, dt, num_iterations,
                                            initial_temp_cont=0.0, nb_flipping=1, cluster_threshold=1.0, init_cluster_size=1, end_cluster_size=1)
        energies.append(energy)
        points.append((sigma.copy(), energy))
        if energy < prev_energy:
            prev_energy = energy
            prev_sigma = sigma.copy()
            best_sigmas.append(prev_sigma.copy())
            if len(best_sigmas) > 10:
                best_sigmas.pop(0)
            # LOGGER.info(f"Current best energy: {prev_energy}")
        if len(points) > 15:
            points.pop(0)
        cluster_size = size_func(i)
        # LOGGER.info(f"Amount of seeds: {int(cluster_size / cluster_size_end)}")
        if cluster_choice=="random":
            cluster = np.random.choice(model.num_variables, size=(cluster_size,), replace=False)
        elif cluster_choice =="median":
            cluster = find_cluster_mean(best_sigmas, prev_sigma, cluster_size, cluster_threshold, 1)
        elif cluster_choice == "energy_approximation":
            cluster = smoothening_cluster(points, cluster_size)
        else:
            cluster = find_cluster_gradient(model, prev_sigma, cluster_size, cluster_threshold)
        sigma = prev_sigma.copy()
        sigma[cluster] *= -1
    LOGGER.info(f"Init cluster size: {cluster_size_init}, final cluster size: {cluster_size_end}, change in size: {size_change}, threshold: {cluster_threshold} Best energy: {prev_energy}")
    return energies, prev_energy, prev_sigma

def do_flipping_local(init_size:list[int], end_size:list[int], sigma_init:np.ndarray, threshold:float, model:IsingModel, nb_flipping:int, dt:float, num_iterations:int, cluster_choice:str):
    results = {}
    for i in range(len(init_size)):
        init_size_i = init_size[i]
        end_size_i = end_size[i]
        sigma_init_i = sigma_init[:, i]
        results[(init_size_i, end_size_i, i)] = (do_flipping(init_size_i, end_size_i, sigma_init_i, threshold, model, nb_flipping, dt, num_iterations, cluster_choice))
    return results

def TSP_flipping():
    graph, best_found = MaxCutParser.G_parser(TOP / "ising/benchmarks/G/G1.txt")
    model = MaxCutParser.generate_maxcut(graph)
    LOGGER.info(f"Best found: {best_found}")

    dt = 1e-4
    num_iterations = 50000
    nb_runs = 10

    nb_nodes = model.num_variables
    sigma = np.random.uniform(-1, 1, (model.num_variables,nb_runs))
    flipping_length = 130
    cluster_size_init = [int(0.6*nb_nodes), int(0.5*nb_nodes)] #int(0.9*nb_nodes), int(0.8*nb_nodes), int(0.7*nb_nodes), 
    str_size_init = [str(s) for s in cluster_size_init]
    cluster_size_end = [int(0.5*nb_nodes), int(0.4*nb_nodes), int(0.3*nb_nodes), int(0.2*nb_nodes), int(0.1*nb_nodes)] #  
    str_size_end = [str(s) for s in cluster_size_end]
    threshold = 1.0
    nb_tasks = len(cluster_size_end)*len(cluster_size_init)*nb_runs
    tasks_per_core = int(nb_tasks // (int(AMOUNT_CORES)/3))
    tasks = [(init_size, final_cluster_size, sigma[:,j], threshold) for init_size in cluster_size_init for final_cluster_size in cluster_size_end for j in range(nb_runs)]
    new_tasks = []
    for task in tasks:
        if len(new_tasks) ==  0:
            new_tasks.append([task])
        else:
            if len(new_tasks[-1]) < tasks_per_core:
                new_tasks[-1].append(task)
            else:
                new_tasks.append([task])
    for i, task in enumerate(new_tasks):
        init_sizes = [t[0] for t in task]
        end_sizes = [t[1] for t in task]
        sigmas = np.array([t[2] for t in task]).T
        new_tasks[i] = (init_sizes, end_sizes, sigmas, threshold)
    with multiprocessing.Pool(processes=tasks_per_core, initializer=os.nice, initargs=(NICENESS,)) as pool:
        flipping_partial = partial(do_flipping_local, model=model, 
                                                nb_flipping=flipping_length, 
                                                dt=dt, 
                                                num_iterations=num_iterations,
                                                cluster_choice="energy_approximation")
        results = pool.starmap(flipping_partial, new_tasks)
    energies = {key: result[1] for subresult in results for key, result in subresult.items()}
    energies = np.array([energies[(init_size, end_size, run)] for init_size in cluster_size_init
                                                              for end_size in cluster_size_end
                                                              for run in range(nb_runs)]).reshape((len(cluster_size_init), len(cluster_size_end), nb_runs))
    best_energies = np.mean(energies, axis=2).reshape((len(cluster_size_init), len(cluster_size_end)))
    np.savetxt("Energy_approximation.pkl", best_energies)
    plot_data(best_energies, "TSP_cluster_size_change.png",  "Final cluster size", str_size_end, "Initial cluster size",str_size_init, best_found)

if __name__ == "__main__":
    TSP_flipping()
