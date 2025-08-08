import numpy as np
import multiprocessing as mp
from functools import partial
import os

from ising.flow import TOP, LOGGER
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP
from ising.solvers.Multiplicative import Multiplicative
from ising.under_dev.Flipping.frozen_gradient_cluster import find_frozen_gradient_cluster
from ising.under_dev.Flipping.gradient_cluster import find_cluster_gradient, find_cluster_gradient_largest
from ising.utils.flow import return_rx  

def option1(model, sigma:np.ndarray, init_size:int, end_size:int, threshold:float, nb_flipping:int, **hyperparameters):
    sigma = sigma.copy()
    energies = []
    size_func = lambda x: int((return_rx(nb_flipping, init_size, end_size)**(x*3)) * (init_size-end_size) + end_size)

    
    prev_energy = np.inf
    prev_sigma = sigma.copy()
    for i in range(nb_flipping):
        sigma, energy = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)

        energies.append(energy)
        if energy < prev_energy:
            prev_energy = energy
            prev_sigma = sigma.copy()
        # LOGGER.info(f"Iteration {i}, energy: {energy}, best energy: {prev_energy}")

        cluster_size = size_func(i)
        cluster = find_frozen_gradient_cluster(model, prev_sigma, 2, cluster_size)
        sigma = prev_sigma.copy()
        sigma[cluster] *= -1
    LOGGER.info(f"Option 1: best energy: {prev_energy}")
    return prev_energy, prev_sigma, energies


def option2(model, sigma, init_size, end_size, threshold, nb_flipping, **hyperparameters):
    sigma = sigma.copy()
    energies = []
    size_func = lambda x: int((return_rx(nb_flipping, init_size, end_size)**(x*3)) * (init_size-end_size) + end_size)

    prev_energy = np.inf
    prev_sigma = sigma.copy()
    for _ in range(nb_flipping):
        if _ == 0:
            sigma, energy = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)
        else:
            sigma, _ = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)
            hyperparameters['frozen_nodes'] = None
            sigma, energy = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)

        energies.append(energy)
        if energy < prev_energy:
            prev_energy = energy
            prev_sigma = sigma.copy()
        cluster_size = size_func(_)
        cluster = find_cluster_gradient(model, prev_sigma, cluster_size, threshold)
        hyperparameters['frozen_nodes'] = cluster
        sigma = prev_sigma.copy()
        sigma[cluster] *= -1
    LOGGER.info(f"Option 2: best energy: {prev_energy}")

    return prev_energy, prev_sigma, energies

def option3(model, sigma, init_size, end_size, threshold, nb_flipping, **hyperparameters):
    sigma = sigma.copy()
    energies = []
    size_func = lambda x: int((return_rx(nb_flipping, init_size, end_size)**(x*3)) * (init_size-end_size) + end_size)

    cluster = []
    prev_energy = np.inf
    prev_sigma = sigma.copy()
    for _ in range(nb_flipping):
        if _ == 0:
            sigma, energy = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)
        else:
            sigma, _ = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)
            hyperparameters['frozen_nodes'] = np.array([x for x in range(model.num_variables) if x not in cluster])
            sigma, _ = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)
            hyperparameters['frozen_nodes'] = None
            sigma, energy = Multiplicative().solve(model=model, initial_state=sigma, **hyperparameters)

        energies.append(energy)
        if energy < prev_energy:
            prev_energy = energy
            prev_sigma = sigma.copy()
        cluster_size = size_func(_)
        cluster = find_cluster_gradient(model, sigma, cluster_size, threshold)
        hyperparameters['frozen_nodes'] = cluster
        sigma = prev_sigma.copy()
        sigma[cluster] *= -1
    LOGGER.info(f"Option 3: best energy: {prev_energy}")

    return prev_energy, prev_sigma, energies

def main(chosen_option="1"):
    graph, best_found = TSP_parser(TOP / 'ising/benchmarks/TSP/burma14.tsp')
    LOGGER.info(f"Best found: {best_found}")
    model = TSP(graph, 1.2)
    np.random.seed(1234)

    niceness = 1
    nb_runs = 10
    threshold = 0.8
    init_size = int(7/8*model.num_variables)
    end_size = int(1/14*model.num_variables)
    nb_flipping = 130

    hyperparameters = {'num_iterations': 60000, 'dtMult':1e-7, 'initial_temp_cont':0.0}
    option = {"1": option1, "2": option2, "3": option3}

    tasks = [(model, np.random.choice([-1, 1], size=(model.num_variables,))) for _ in range(nb_runs)]

    with mp.Pool(initializer=os.nice, initargs=(niceness,)) as pool:
        partial_option1 = partial(option[chosen_option], 
                                  init_size=init_size,
                                  end_size=end_size, 
                                  threshold=threshold,
                                  nb_flipping=nb_flipping,
                                  **hyperparameters)
        results1 = pool.starmap(partial_option1, tasks)
        LOGGER.info(f"Finished option {chosen_option}")


    best_energies1 = np.array([result[0] for result in results1])

    summary1 = np.array([np.min(best_energies1), np.max(best_energies1), np.mean(best_energies1), np.std(best_energies1)])

    np.savetxt(TOP / f"ising/under_dev/Flipping/burma14_summary_option{chosen_option}.csv", summary1, fmt="%.2f", header="min max avg std")
    np.savetxt(TOP / f"ising/under_dev/Flipping/burma14_energies_option{chosen_option}.csv", best_energies1, fmt="%.2f", header="energies")

if __name__ == "__main__":
    main("2")