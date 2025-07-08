import numpy as np
import multiprocessing as mp
from functools import partial

from ising.under_dev.Flipping.new_strategy import do_flipping
from ising.flow import TOP, LOGGER

from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP

save_path = TOP / "ising/under_dev/Flipping"

def run_benchmark(benchmark_name:str)->None:
    graph, _ = TSP_parser(TOP / f"ising/benchmarks/TSP/{benchmark_name}.tsp")
    model = TSP(graph, 1.2)

    nb_runs = 100
    threshold = 0.7
    init_size = int(7/8*model.num_variables)
    end_size = int(1/12*model.num_variables)
    nb_flipping = 100
    dt = 1e-6
    num_iterations = 50000
    sigma = np.random.choice([-1, 1], size=(model.num_variables, nb_runs))

    tasks = [(init_size, end_size, sigma[:, i], threshold) for i in range(nb_runs)]
    with mp.Pool(int(mp.cpu_count()/3)) as pool:
        flipping_partial = partial(do_flipping, model=model, 
                                   nb_flipping=nb_flipping, 
                                   dt=dt, 
                                   num_iterations=num_iterations,
                                   cluster_choice="random")
        results = pool.starmap(flipping_partial, tasks)
    # for run in range(nb_runs):
    #     LOGGER.info(f"RUN {run + 1}/{nb_runs}")
    #     _, best_energy, _ = do_flipping(model, sigma, init_size, end_size, nb_flipping, dt, num_iterations, threshold)
    #     all_energies.append(best_energy)
    all_energies = np.array([result[1] for result in results])
    summary = np.array([[np.min(all_energies),np.max(all_energies),np.mean(all_energies),np.std(all_energies)]])
    header = "min max avg std"
    np.savetxt(save_path / f"{benchmark_name}_summary_random_flipping.csv", summary, fmt="%.2f", header=header)

if __name__ == "__main__":
    # run_benchmark("burma14")
    # run_benchmark("ulysses16")
    run_benchmark("ulysses22")
