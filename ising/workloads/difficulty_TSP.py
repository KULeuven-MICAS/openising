import tsplib95
import numpy as np
import networkx as nx

from ising.stages import TOP
from ising.generators.TSP import TSP
from ising.solvers.Multiplicative import Multiplicative
benchmark_top = TOP / "ising/benchmarks/TSP"
benchmarks = ["burma14.tsp", "ulysses16.tsp", "ulysses22.tsp", "bayg29.tsp", "bays29.tsp"]
best_energies = [3323, 6859, 7013, 1610, 2020]
nb_runs = 10

for benchmark_name, best_en in zip(benchmarks, best_energies):
    benchmark = str(benchmark_top / benchmark_name)

    problem = tsplib95.load(benchmark)

    Cost = nx.linalg.adjacency_matrix(problem.get_graph()).toarray()
    model = TSP(graph=problem.get_graph(), weight_constant=1.2)
    variance = np.std(Cost)
    print(f"Benchmark: {benchmark_name}, variance: {variance}")
    print(f"cond(Cost): {np.linalg.cond(Cost)}, spectrum(Cost): {np.linalg.eigvals(Cost)}")
    print(f"cond(pseudoinv(Cost)):{np.linalg.cond(np.linalg.pinv(Cost))}, cond(J): {np.linalg.cond(model.J)}")
    accs = []
    for run in range(nb_runs):
        initial_state = np.random.choice([-1, 1], size=model.num_variables)
        sol_state, sol_en, _, _, _ = Multiplicative().solve(model, initial_state, 50000, 100, 0, 0.9, 0.01)
        acc = np.abs(sol_en - best_en) / np.abs(best_en)
        accs.append(acc)
    print(f"Benchmark: {benchmark_name}, mean acc: {np.mean(accs)}, std acc: {np.std(accs)}")



