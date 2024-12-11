import numpy as np
import os
import argparse
import pathlib

from ising.generators.MaxCut import MaxCut
from ising.benchmarks.parsers.G import G_parser
from ising.solvers.SB import discreteSB, ballisticSB
from ising.postprocessing.energy_plot import plot_energies_multiple

parser = argparse.ArgumentParser()
parser.add_argument('-benchmark', help="Which Max-Cut benchmark to generate", default='G_dummy.txt')
parser.add_argument('-nb_runs', help='Number of runs', default=1)
parser.add_argument('-num_iter', help='range of total amount of iterations', default=(100, 1000))
parser.add_argument('-it_step', help='Step between number of iterations', default=100)
parser.add_argument('-dt', help='Time step', default=0.25)
args = parser.parse_args()
TOP = pathlib.Path(os.getenv("TOP"))

print("Generating G benchmark...")
benchmark = TOP / 'ising/benchmarks/G'
graph_orig = G_parser(benchmark=benchmark / args.benchmark)
model = MaxCut(graph=graph_orig)
print("Done generating G benchmark")
print("Generated the following model:\n", model)

logfile_top = TOP / "ising/flow/logs"
print("Setting up parameters...")
x = np.random.uniform(-0.1, 0.1, (model.num_variables,))
y = np.zeros((model.num_variables,))
a0 = 1.
c0 = 0.5 / (np.sqrt(model.num_variables) * \
            np.sqrt(np.sum(np.power(model.J, 2))/(model.num_variables*(model.num_variables-1))))

list_num_iter = list(range(args.num_iter[0], args.num_iter[1], args.it_step))
logfiles = []
for i in range(int(args.nb_runs)):
    print(f"Run {i+1}/{args.nb_runs}")
    for num_iterations in list_num_iter:
        print(f"Number of iterations: {num_iterations}")
        def at(t):
            return a0 / (args.dt * num_iterations) * t
        print("Solving with discreteSB...")
        logfile = logfile_top / f"discreteSB_it{num_iterations}_run{i}.log"
        logfiles.append(logfile)
        state, energy = discreteSB().solve(model, x, y, num_iterations, at, c0, args.dt, a0, file=logfile)
        print(f"  state={state}, energy={energy}")

        print("Solving with ballisticSB...")
        logfile = logfile_top / f"ballisticSB_it{num_iterations}_run{i}.log"
        logfiles.append(logfile)
        state, energy = ballisticSB().solve(model, x, y, num_iterations, at, c0, a0, args.dt, file=logfile)
        print(f"  state={state}, energy={energy}")

plot_energies_multiple(logfiles, save_folder=TOP / 'ising/flow/plots')
