import os
import pathlib
import numpy as np
import argparse

from ising.generators.MaxCut import MaxCut
from ising.benchmarks.parsers.G import G_parser
from ising.solvers.BRIM import BRIM
from ising.postprocessing.energy_plot import plot_energies_multiple
from ising.postprocessing.plot_solutions import plot_state_continuous

# from ising.postprocessing.MC_plot import plot_MC_solution

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
parser.add_argument("-tend", help="End time for the simulation", default=3e-5)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("--G_range", help="Range of resistor parameter", default=(0.01, 50.0), nargs="+")
parser.add_argument("--kmin_range", help="Range of minimum latch strength", default=(0.01, 5.0), nargs="+")
parser.add_argument("-kmax", help="Maximum latch strength", default=5.0)
parser.add_argument("-nb_runs", help="Number of runs", default=5)

args = parser.parse_args()
logfile_top = TOP / "ising/flow/logs"
figure_folder = TOP / "ising/flow/plots/BLIM_benchmark_test"
dt = float(args.tend) / float(args.num_iter)


benchmark = TOP / "ising/benchmarks/G/G1.txt"
graph_g = G_parser(benchmark)
problem = MaxCut(graph_g)
best_found = -11624.0
kmin_range = tuple(args.kmin_range)
v = np.random.choice([-0.5, 0.5], (problem.num_variables,))
k_min_list = np.linspace(float(kmin_range[0]), float(kmin_range[1]), int(args.nb_runs))
logfiles = []
for kmin in k_min_list:
    print(f"current minimum latch strength: {kmin}")
    logfile = logfile_top / f"BRIM_G1benchmark_kmin{kmin}.log"
    state_energy = BRIM().solve(
        model=problem,
        num_iterations=int(args.num_iter),
        v=v,
        dt=dt,
        kmin=kmin,
        kmax=float(args.kmax),
        C=float(args.C),
        G=float(args.G),
        file=logfile,
    )
    plot_state_continuous(logfile, f"BRIM_G1benchmark_state_kmin{kmin}.png", save_folder=figure_folder)
    logfiles.append(logfile)

plot_energies_multiple(
    logfiles, figName="BLIM_kmin_comparison.png", best_found=best_found, save_folder=figure_folder, diff_metadata="kmin"
)
G_range = tuple(args.G_range)
G_list = np.linspace(float(G_range[0]), float(G_range[1]), int(args.nb_runs))
logfiles = []
for G in G_list:
    print(f"current resistor strength: {G}")
    logfile = logfile_top / f"BRIM_G1benchmark_G{G}.log"
    state_energy = BRIM().solve(
        model=problem,
        num_iterations=int(args.num_iter),
        v=v,
        dt=dt,
        kmin=k_min_list[0],
        kmax=float(args.kmax),
        C=float(args.C),
        G=G,
        file=logfile,
    )
    logfiles.append(logfile)
    plot_state_continuous(logfile, f"BRIM_G1benchmark_state_G{G}.png", save_folder=figure_folder)
plot_energies_multiple(
    logfiles, figName="BLIM_G_comparison.png", best_found=best_found, save_folder=figure_folder, diff_metadata="G"
)

