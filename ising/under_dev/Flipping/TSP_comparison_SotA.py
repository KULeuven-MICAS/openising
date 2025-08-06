import numpy as np
import multiprocessing as mp
from functools import partial
import os

from ising.stages.model.ising import IsingModel
from ising.under_dev.Flipping.new_strategy import do_flipping
from ising.flow import TOP, LOGGER

from ising.under_dev import MaxCutParser
from ising.benchmarks.parsers.Knapsack import QKP_parser
from ising.generators.Knapsack import knapsack
from ising.benchmarks.parsers.TSP import TSP_parser
from ising.generators.TSP import TSP

save_path = TOP / "ising/under_dev/Flipping/9M_results"

def run_benchmark_TSP(benchmark_name:str)->None:
    graph, _ = TSP_parser(TOP / f"ising/benchmarks/TSP/{benchmark_name}.tsp")
    model = TSP(graph, 1.2)
    hyperparameters = {"dt": 1e-6, "num_iterations": 50000, "init_size":7/8, "end_size":1/12, "threshold":0.8, "cluster_choice":"gradient"}
    run_benchmark(model, benchmark_name, **hyperparameters)

def run_benchmark_MCP(benchmark_name:str)->None:
    g, _ = MaxCutParser.G_parser(TOP / f"ising/benchmarks/G/{benchmark_name}.txt")
    model = MaxCutParser.generate_maxcut(g)
    hyperparameters = {"dt": 1e-4, "num_iterations": 100000, "init_size":1, "end_size":1/50, "threshold":0.7, "cluster_choice":"random"}
    run_benchmark(model, benchmark_name, **hyperparameters)

def run_benchmark_QKP(benchmark_name:str)->None:
    profit, weight, capacity, _ = QKP_parser(TOP / f"ising/benchmarks/Knapsack/{benchmark_name}.txt")
    model = knapsack(profit, capacity, weight, 1.2)
    hyperparameters = {"dt": 1e-6, "num_iterations": 50000, "init_size":1/2, "end_size":1/100, "threshold":0.7, "cluster_choice":"gradient"}
    run_benchmark(model, benchmark_name, **hyperparameters)


def run_benchmark(model: IsingModel, benchmark_name:str, **hyperparameters) -> None:
    nb_runs = 100
    threshold = hyperparameters["threshold"]
    init_size = int(hyperparameters["init_size"]*model.num_variables)
    end_size = int(hyperparameters["end_size"]*model.num_variables)
    nb_flipping = 130
    sigma = np.random.choice([-1, 1], size=(model.num_variables, nb_runs))

    tasks = [(init_size, end_size, sigma[:, i], threshold) for i in range(nb_runs)]
    with mp.Pool(initializer=os.nice, initargs=(1,)) as pool:
        flipping_partial = partial(do_flipping, model=model, 
                                   nb_flipping=nb_flipping,
                                   dt=hyperparameters["dt"],
                                   num_iterations=hyperparameters["num_iterations"],
                                   cluster_choice=hyperparameters["cluster_choice"],)
        results = pool.starmap(flipping_partial, tasks)

    all_energies = np.array([result[1] for result in results])
    summary = np.array([[np.min(all_energies),np.max(all_energies),np.mean(all_energies),np.std(all_energies)]])
    header = "min max avg std"
    np.savetxt(save_path / f"{benchmark_name}_summary_{hyperparameters["cluster_choice"]}_flipping.csv", summary, fmt="%.2f", header=header)
    np.savetxt(save_path / f"{benchmark_name}_energies_{hyperparameters["cluster_choice"]}_flipping.pkl", all_energies, fmt="%.2f")

def parse_out_file(filename: str, save_file_name:str):
    results = []
    with open(filename, 'r') as file:
        for line in file:
            stripped_line = line.split(" ")
            if stripped_line[0] == "INFO:Threshold:":
                results.append(float(stripped_line[-1][:-2]))
    np.savetxt(save_path / f"{save_file_name}.pkl", results)

if __name__ == "__main__":
    # run_benchmark_TSP("burma14")
    run_benchmark_MCP("K2000")
    # run_benchmark_QKP("jeu_100_25_1")
    # parse_out_file(TOP / "ising/under_dev/Flipping/burma14_comparison_random.out", "TSP_random_flipping")