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
from ising.under_dev import TSPParser
from ising.generators.TSP import TSP
from ising.generators.MIMO import MIMO_to_Ising, MU_MIMO

save_path = TOP / "ising/under_dev/Flipping"

def run_benchmark_TSP(benchmark_name:str)->None:
    graph, _ = TSPParser.TSP_parser(TOP / f"ising/benchmarks/TSP/{benchmark_name}.tsp")
    model = TSP(graph, 1.2)
    hyperparameters = {"dt": 1e-6, "num_iterations": 50000, "init_size":7/8, "end_size":1/12, "threshold":0.3, "cluster_choice":"frequency"}
    run_benchmark(model, benchmark_name, **hyperparameters)

def run_benchmark_MCP(benchmark_name:str)->None:
    g, _ = MaxCutParser.G_parser(TOP / f"ising/benchmarks/G/{benchmark_name}.txt")
    model = MaxCutParser.generate_maxcut(g)
    hyperparameters = {"dt": 1e-4, "num_iterations": 100000, "init_size":1, "end_size":1/50, "threshold":0.3, "cluster_choice":"frequency"}
    run_benchmark(model, benchmark_name, **hyperparameters)

def run_benchmark_QKP(benchmark_name:str)->None:
    profit, weight, capacity, _ = QKP_parser(TOP / f"ising/benchmarks/Knapsack/{benchmark_name}.txt")
    model = knapsack(profit, capacity, weight, 1.2)
    hyperparameters = {"dt": 1e-6, "num_iterations": 50000, "init_size":1/2, "end_size":1/100, "threshold":0.3, "cluster_choice":"frequency"}
    run_benchmark(model, benchmark_name, **hyperparameters)

def run_benchmark_MIMO(SNR, M, N):
    H, symbols = MU_MIMO(N, N, M, 1)
    hyperparameters = {"dt":1e-5, "num_iterations":50000, "init_size":7/8, "end_size":1/16, "threshold": 0.3, "cluster_choice":"frequency", "H": H, "M":M, "symbols": symbols, "SNR": SNR, "N":N}
    
    run_benchmark(MIMO_to_Ising(H, np.random.choice(symbols, size=(N,)) + 1j*np.random.choice(symbols, size=(N,)), SNR, N, N, M, 1)[0], "MIMO", **hyperparameters)

def do_flipping_local(init_size:int, end_size:int, sigma_init:np.ndarray, threshold:float, model:IsingModel, nb_flipping:int, dt:float, num_iterations:int, cluster_choice:str):
    nb_trials = np.shape(sigma_init)[1]
    results = []
    for i in range(nb_trials):
        if isinstance(model, IsingModel):
            results.append(do_flipping(init_size, end_size, sigma_init[:, i], threshold, model, nb_flipping, dt, num_iterations, cluster_choice))
        else:
            results.append(do_flipping(init_size, end_size, sigma_init[:, i], threshold, model[i], nb_flipping, dt, num_iterations, cluster_choice))
    return results

def run_benchmark(model: IsingModel, benchmark_name:str, **hyperparameters) -> None:
    nb_runs = 100
    threshold = hyperparameters["threshold"]
    init_size = int(hyperparameters["init_size"]*model.num_variables)
    end_size = int(hyperparameters["end_size"]*model.num_variables)
    nb_flipping = 130
    nb_cores = 12
    step = nb_runs // nb_cores
    sigma = np.random.choice([-1, 1], size=(model.num_variables, nb_runs))
    if benchmark_name == "MIMO":
        N = hyperparameters["N"]
        xt1 = np.random.choice(hyperparameters["symbols"], size=(N,nb_runs)) + 1j*np.random.choice(hyperparameters["symbols"], 
                                                                                                                      size=(N,nb_runs))
        models = [MIMO_to_Ising(hyperparameters["H"], xt1[:, i], hyperparameters["SNR"], N, N, hyperparameters["M"], 1)[0] for i in range(nb_runs)]
        tasks = [(init_size, end_size, sigma[:, i:i+step], threshold, models[i:i+step]) for i in range(0, nb_runs, step)]
    else:
        tasks = [(init_size, end_size, sigma[:, i:i+step], threshold, model) for i in range(0, nb_runs, step)]
    # do_flipping(init_size, end_size, sigma[:, 0], threshold, model, nb_flipping,hyperparameters["dt"],  hyperparameters["num_iterations"])
    
    with mp.Pool(processes=nb_cores, initializer=os.nice, initargs=(1,)) as pool:
        flipping_partial = partial(do_flipping_local,
                                   nb_flipping=nb_flipping,
                                   dt=hyperparameters["dt"],
                                   num_iterations=hyperparameters["num_iterations"],
                                   cluster_choice=hyperparameters["cluster_choice"],)
        results = pool.starmap(flipping_partial, tasks)

    all_energies = np.empty((nb_runs,))
    for part, subresult in enumerate(results):
        for i, result in enumerate(subresult):
            all_energies[part*step + i] = result[1]
    # all_energies = np.array([result[1] for result in subresult for subresult in results])
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
    # run_benchmark_MIMO(5, 16, 5)
    run_benchmark_MCP("G11")
    # run_benchmark_QKP("jeu_100_25_1")
    # parse_out_file(TOP / "ising/under_dev/Flipping/burma14_comparison_random.out", "TSP_random_flipping")
