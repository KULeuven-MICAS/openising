import numpy as np
import os
import argparse
import pathlib

from ising.generators.MaxCut import MaxCut
from ising.benchmarks.parsers.G import G_parser
from ising.solvers.SB import discreteSB, ballisticSB
from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers
from ising.postprocessing.plot_solutions import plot_state_continuous

parser = argparse.ArgumentParser()
parser.add_argument("-benchmark", help="Which Max-Cut benchmark to generate", default="G_dummy.txt")
parser.add_argument("-nb_runs", help="Number of runs", default=2)
parser.add_argument("--num_iter", help="range of total amount of iterations", default=(100, 1000), nargs="+", type=int)
#parser.add_argument("-it_step", help="Step between number of iterations", default=100)
parser.add_argument("-dt", help="Time step", default=0.25)

print("Parsing args")
args = parser.parse_args()
nb_runs = int(args.nb_runs)
num_iter = tuple(args.num_iter)
#it_step = int(args.it_step)
dt = float(args.dt)
benchmark = str(args.benchmark)
TOP = pathlib.Path(os.getenv("TOP"))

print("Generating G benchmark...")
benchmarktop = TOP / "ising/benchmarks/G"
graph_orig = G_parser(benchmark=benchmarktop / benchmark)
model = MaxCut(graph=graph_orig)
print("Done generating G benchmark")
print("Generated the following model:\n", model)

logfile_top = TOP / "ising/flow/logs"
figtop = TOP / "ising/flow/plots/SB_comparison"

print("Setting up parameters...")
x = np.random.uniform(-0.1, 0.1, (model.num_variables,))
y = np.zeros((model.num_variables,))
a0 = 1.0
c0 = 0.5 / (
    np.sqrt(model.num_variables)
    * np.sqrt(np.sum(np.power(model.J, 2)) / (model.num_variables * (model.num_variables - 1)))
)
list_num_iter = np.linspace(num_iter[0], num_iter[1], nb_runs, dtype=int)


logfiles = dict()
for num_iterations in list_num_iter:
    logfiles[num_iterations] = dict()
    logfiles[num_iterations]["dSB"] = []
    logfiles[num_iterations]["bSB"] = []

    print(f"Number of iterations: {num_iterations}")

    def at(t):
        return a0 / (dt * num_iterations) * t

    for i in range(nb_runs):
        print(f"Run {i+1}/{nb_runs}")

        print("Solving with discreteSB...")
        logfile = logfile_top / f"discreteSB_it{num_iterations}_run{i}.log"
        logfiles[num_iterations]["dSB"].append(logfile)
        discreteSB().solve(model, x, y, num_iterations, at, c0, dt, a0, file=logfile)

        print("Solving with ballisticSB...")
        logfile = logfile_top / f"ballisticSB_it{num_iterations}_run{i}.log"
        logfiles[num_iterations]["bSB"].append(logfile)
        ballisticSB().solve(model, x, y, num_iterations, at, c0, a0, dt, file=logfile)
    plot_state_continuous(
        logfile=logfiles[num_iterations]["dSB"][0], figname=f"dSB_state_it{num_iterations}.png", save_folder=figtop
    )
    plot_state_continuous(
        logfile=logfiles[num_iterations]["bSB"][0], figname=f"bSB_state_it{num_iterations}.png", save_folder=figtop
    )

plot_energy_dist_multiple_solvers(
    logfiles, figName="SB_comparison_it_length.png", xlabel="Iteration length", save_folder=figtop
)
