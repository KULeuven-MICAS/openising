import os
import pathlib
import numpy as np
import argparse
import openjij as oj

from ising.generators.MaxCut import random_MaxCut, MaxCut
from ising.benchmarks.parsers.G import G_parser
from ising.solvers.BRIM import BRIM
from ising.postprocessing.energy_plot import plot_energy_accuracy_check, plot_energies
from ising.postprocessing.plot_solutions import plot_state_continuous

# from ising.postprocessing.MC_plot import plot_MC_solution
from ising.utils.numpy import triu_to_symm

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument("-Nlist", help="tuple containing min and max problem size", default=(10, 100))
parser.add_argument("-Njump", help="The jump between two problem sizes", default=10)
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
parser.add_argument("-tend", help="End time for the simulation", default=3e-5)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("-kmin", help="Minimum latch strength", default=5.0)
parser.add_argument("-kmax", help="Maximum latch strength", default=5.0)
parser.add_argument("-nb_runs", help="Number of runs", default=2)

args = parser.parse_args()
dt = float(args.tend) / float(args.num_iter)
logfile_top = TOP / "ising/flow/logs"
figure_folder = TOP / "ising/flow/plots/BLIM_test"
logfiles = dict()
best_found_list = []
for N in range(int(args.Nlist[0]), int(args.Nlist[1]), int(args.Njump)):
    print(f"Generating random problem of size {N}...")
    problem = random_MaxCut(N)
    print("the generated problem: ", problem)
    print("Done generating random problem")

    bqm = oj.BinaryQuadraticModel.from_numpy_matrix(np.diag(problem.h) - triu_to_symm(problem.J))
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=int(args.nb_runs))
    best_found = response.first.energy
    print("Best OpenJij solution: ", best_found)
    print("best OpenJij state:", response.first.sample)
    best_found_list.append(best_found)

    v = np.random.choice([-1, 1], (N,))
    logfiles[N] = []
    for i in range(int(args.nb_runs)):
        print(f"Run {i+1}/{args.nb_runs}")
        logfile = logfile_top / f"BRIM_N{N}_run{i}.log"
        state, energy = BRIM().solve(
            model=problem,
            num_iterations=int(args.num_iter),
            v=v,
            dt=dt,
            kmin=float(args.kmin),
            kmax=float(args.kmax),
            C=float(args.C),
            G=float(args.G),
            file=logfile,
        )
        # print(f"state={state}, energy={energy}")
        logfiles[N].append(logfile)
    plot_state_continuous(logfile, f"BRIM_N{N}_state.png", save_folder=figure_folder)
plot_energy_accuracy_check(
    logfiles, figName="BRIM_energy_dist.png", best_found=best_found_list, save_folder=figure_folder
)

print("Testing on G1 benchmark")

benchmark = TOP / "ising/benchmarks/G/G1.txt"
graph_g = G_parser(benchmark)
problem = MaxCut(graph_g)
best_found = -11624.0

v = np.random.choice([-0.5, 0.5], (problem.num_variables,))
logfile = logfile_top / "BRIM_G1benchmark.log"
state_energy = BRIM().solve(
    model=problem,
    num_iterations=int(args.num_iter),
    v=v,
    dt=dt,
    kmin=float(args.kmin),
    kmax=float(args.kmax),
    C=float(args.C),
    G=float(args.G),
    file=logfile,
)
plot_energies(fileName=logfile, figName="BRIM_G1benchmark.png", best_found=best_found, save_folder=figure_folder)
plot_state_continuous(logfile, "BRIM_G1benchmark_state.png", save_folder=figure_folder)
