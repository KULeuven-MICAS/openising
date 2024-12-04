import os
import pathlib
import numpy as np
import argparse
import openjij as oj

from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import random_MaxCut
from ising.solvers.BRIM import BRIM
from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers
from ising.postprocessing.MC_plot import plot_MC_solution

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument('-Nlist', help="tuple containing min and max problem size", default=(10, 100))
parser.add_argument('-Njump', help="The jump between two problem sizes", default=10)
parser.add_argument('-num_iter', help="Number of iterations for each run", default=1000)
parser.add_argument('-tend', help="End time for the simulation", default=3e-5)
parser.add_argument('-C', help="capacitor parameter", default=1e-5)
parser.add_argument('-G', help="Resistor parameter", default=1e-1)
parser.add_argument('-kmin', help='Minimum latch strength', default=5.)
parser.add_argument('-kmax', help='Maximum latch strength', default=5.)
parser.add_argument('-nb_runs', help='Number of runs', default=2)

args = parser.parse_args()
logfile_top = TOP / "ising/flow/logs"
for N in range(args.Nlist[0], args.Nlist[1], args.Njump):
    print("Generating random problem...")
    problem = random_MaxCut(N)
    print("Done generating random problem")

    v = np.random.uniform()
    for i in range(args.nb_runs):
        print(f"Run {i+1}/{args.nb_runs}")
        print("Solving with BRIM...")
        logfile = logfile_top / f"BRIM_N{N}_run{i}.log"
        state, energy = BRIM().solve(problem, args.num_iter, args.tend, args.C, args.G, args.kmin, args.kmax, file=logfile)
        print(f"  state={state}, energy={energy}")

