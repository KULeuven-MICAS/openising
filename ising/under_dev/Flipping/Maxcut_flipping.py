import numpy as np
import multiprocessing
from functools import partial
import os

from ising.flow import TOP, LOGGER
from ising.under_dev.Flipping.new_strategy import do_flipping
from ising.under_dev.Flipping.plotting import plot_data, figtop
from ising.under_dev import MaxCutParser
from ising.solvers.Multiplicative import Multiplicative
from ising.postprocessing.summarize_energies import summary_energies

NICENESS = 1

def MaxCut_flipping():
    g, best_found = MaxCutParser.G_parser(TOP / "ising/benchmarks/G/K2000.txt")
    model = MaxCutParser.generate_maxcut(g)
    LOGGER.info(f"Best found energy: {best_found:.2f}")

    # Cluster parameters
    init_cluster_size = int(7/8*model.num_variables)
    final_cluster_size = int(1/12*model.num_variables)
    nb_flipping = 100
    threshold = 0.8
    nb_runs = 100
    sigmas = np.random.choice([-1, 1], size=(model.num_variables, nb_runs))

    # Solver parameters
    dt = 1e-4
    num_iterations = 100000

    # solutions_original = []
    _, sol_en = Multiplicative().solve(model, sigmas[:, 0], dtMult=dt, num_iterations=200000, initial_temp_cont=0.0)
    LOGGER.info(f"Original solution energy: {sol_en:.2f}")

    # best_energies = np.zeros((len(cluster_sizes), len(nb_flipping)))
    tasks = [(init_cluster_size, final_cluster_size, sigmas[:, i]) for i in range(nb_runs)]

    with multiprocessing.Pool(initializer=os.nice, initargs=(NICENESS,)) as pool:
        partial_flipping = partial(do_flipping, model=model, dt=dt, num_iterations=num_iterations, cluster_threshold=threshold, nb_flipping=nb_flipping)
        results = pool.starmap(partial_flipping, tasks)
    best_energies = np.array((result[1] for result in results)).reshape((nb_runs,))
    np.savetxt(TOP / "ising/under_dev/Flipping/maxcut_flipping_results.csv", [np.average(best_energies), np.min(best_energies), np.max(best_energies), np.std(best_energies)],
               fmt="%.2f", header="avg, min, max, std")

if __name__ == "__main__":
    MaxCut_flipping()
    LOGGER.info("MaxCut flipping completed successfully.")