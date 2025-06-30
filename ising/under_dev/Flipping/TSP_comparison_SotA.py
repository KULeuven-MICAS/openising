import numpy as np

from ising.under_dev.Flipping.new_strategy import do_flipping
from ising.flow import TOP, LOGGER

from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP

save_path = TOP / "ising/under_dev/Flipping"

def run_benchmark(benchmark_name:str)->None:
    graph, _ = TSP_parser(TOP / f"ising/benchmarks/TSP/{benchmark_name}")
    model = TSP(graph, 1.2)

    nb_runs = 100
    threshold = 0.7
    init_size = int(2/3*model.num_variables)
    end_size = int(1/18*model.num_variables)
    nb_flipping = 100
    dt = 1e-6
    num_iterations = 50000

    all_energies = []
    for run in range(nb_runs):
        LOGGER.info(f"RUN {run + 1}/{nb_runs}")
        sigma = np.random.choice([-1, 1], size=(model.num_variables,))
        _, best_energy, _ = do_flipping(model, sigma, init_size, end_size, nb_flipping, dt, num_iterations, threshold)
        all_energies.append(best_energy)
    summary = np.array([[np.min(all_energies),np.max(all_energies),np.mean(all_energies),np.std(all_energies)]])
    header = "min max avg std"
    np.savetxt(save_path / f"{benchmark_name}_summary_flipping.csv", summary, fmt="%.2f", header=header)

if __name__ == "__main__":
    run_benchmark("burma14")
    run_benchmark("ulysses16")
    run_benchmark("ulysses22")
