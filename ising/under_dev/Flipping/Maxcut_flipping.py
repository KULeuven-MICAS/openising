import numpy as np
import multiprocessing
from functools import partial
import os

from ising.flow import TOP, LOGGER
from ising.under_dev.Flipping.new_strategy import do_flipping
from ising.under_dev.Flipping.plotting import plot_data, figtop
from ising.under_dev import MaxCutParser
from ising.solvers.Multiplicative import Multiplicative

NICENESS = 1

def MaxCut_flipping():
    g16, best_found = MaxCutParser.G_parser(TOP / "ising/benchmarks/G/G6.txt")
    model_g1 = MaxCutParser.generate_maxcut(g16)
    LOGGER.info(f"Best found energy: {best_found:.2f}")

    # Cluster parameters
    init_cluster_sizes = [1/2, 3/4, 4/5, 7/8, 1]
    str_sizes_init = ["0.5", "0.75", "0.8", "0.875", "1"]
    final_cluster_sizes = [1/15,1/10, 1/7, 1/5, 1/4 ]
    str_sizes_final = ["1/15","1/10", "1/7", "1/5", "1/4"]
    nb_flipping = 100
    threshold = 0.7
    nb_runs = 10
    sigmas = np.random.choice([-1, 1], size=(model_g1.num_variables, nb_runs))

    sol_state, sol_en = Multiplicative().solve(model_g1, sigmas[:, 0], dtMult=dt, num_iterations=200000, initial_temp_cont=0.0)
    LOGGER.info(f"Original solver found energy: {sol_en:.2f} with state: {sol_state}")
    # Solver parameters
    dt = 1e-4
    num_iterations = 100000
    # best_energies = np.zeros((len(cluster_sizes), len(nb_flipping)))
    tasks = [(init_size, final_size, sigmas[:, i]) for init_size in init_cluster_sizes for final_size in final_cluster_sizes for i in range(nb_runs)]

    with multiprocessing.Pool(initializer=os.nice, initargs=(NICENESS,)) as pool:
        partial_flipping = partial(do_flipping, model=model_g1, dt=dt, num_iterations=num_iterations, cluster_threshold=threshold, nb_flipping=nb_flipping)
        results = pool.starmap(partial_flipping, tasks)
    
    best_energies = np.array([[result[1] for result in results]]).reshape((len(init_cluster_sizes), len(final_cluster_sizes), nb_runs))
    for i, init_size in enumerate(str_sizes_init):
        np.savetxt(figtop / f"G6_flipping_init_{init_size}.pkl", best_energies[i, :, :])
    avg_best_energies = np.mean(best_energies, axis=2)

    plot_data(avg_best_energies, "G16_flipping.png", "final cluster size", str_sizes_final, "init cluster sizes", str_sizes_init, best_found)

if __name__ == "__main__":
    MaxCut_flipping()
    LOGGER.info("MaxCut flipping completed successfully.")