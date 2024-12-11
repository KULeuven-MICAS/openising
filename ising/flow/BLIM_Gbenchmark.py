import os
import pathlib
import numpy as np
import argparse
import openjij as oj

from ising.generators.MaxCut import MaxCut
from ising.benchmarks.parsers.G import G_parser
from ising.solvers.BRIM import BRIM
from ising.postprocessing.energy_plot import plot_energies
from ising.postprocessing.plot_solutions import plot_state_continuous

# from ising.postprocessing.MC_plot import plot_MC_solution

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
parser.add_argument("-tend", help="End time for the simulation", default=3e-5)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("-kmin", help="Minimum latch strength", default=5.0)
parser.add_argument("-kmax", help="Maximum latch strength", default=5.0)
parser.add_argument("-nb_runs", help="Number of runs", default=2)

args = parser.parse_args()
logfile_top = TOP / "ising/flow/logs"
figure_folder = TOP / "ising/flow/plots/BLIM_test"
dt = float(args.tend) / float(args.num_iter)

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
