import numpy as np
import os
import argparse
import pathlib

from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.utils.flow import make_directory, parse_hyperparameters

from ising.utils.helper_solvers import return_c0, return_rx, return_G, return_q
from ising.utils.threading import make_solvers_thread

TOP = pathlib.Path(os.getenv("TOP"))

parser = argparse.ArgumentParser()
parser.add_argument("-benchmark", help="Name of the benchmark to run", default="G1")
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=15)
parser.add_argument("--num_iter", help="Range for number of iterations", default=(500, 5000), nargs="+")

# BRIM parameters
parser.add_argument("-dtBRIM", help="time_step for BRIM", default=3e-9)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("-k_min", help="Minimum latch strength", default=0.01)
parser.add_argument("-k_max", help="Maximum latch strength", default=2.5)
parser.add_argument("-flip", help="Whether to activate random flipping in BRIM", default=True)

# SA parameters
parser.add_argument("-T", help="Initial temperature", default=50.0)
parser.add_argument("-T_final", help="Final temperature of the annealing process", default=0.05)
parser.add_argument("-seed", help="Seed for random number generator", default=0)

# SCA parameters
parser.add_argument("-q", help="initial penalty value", default=0.0)
parser.add_argument("-q_final", help="final penalty value", default=10.0)

# SB parameters
parser.add_argument("-dt", help="Time step for simulated bifurcation", default=0.25)
parser.add_argument("-a0", help="Parameter a0 of SB", default=1.0)
parser.add_argument("-c0", help="Parameter c0 of SB", default=0.0)

args = parser.parse_args()


benchmark = args.benchmark
print("Generating benchmark: ", benchmark)
graph, best_found = G_parser(benchmark=TOP / f"ising/benchmarks/G/{benchmark}.txt")
model = MaxCut(graph=graph)
if best_found is not None:
    print("Best found energy: ", -best_found)
print("Generated benchmark")

if args.solvers == "all":
    solvers = ["SA", "SCA", "bSB", "dSB", "BRIM"]
else:
    solvers = args.solvers[0].split()
print("Solving with following solvers: ", solvers)

num_iter = args.num_iter[0].split()
nb_runs = int(args.nb_runs)
iter_list = np.array(range(int(num_iter[0]),int(num_iter[1]), 50))

print("Setting up solvers")
logpath = TOP / "ising/flow/MaxCut/logs"
make_directory(logpath)

print(np.sum(model.J, axis=1))

for num_iter in iter_list:
    print(f"Running for {num_iter} iterations")
    s_init = np.random.choice([-1, 1], (model.num_variables,))
    hyperparameters = parse_hyperparameters(args, num_iter)

    if hyperparameters["G"] == 0.0:
        hyperparameters["G"] = return_G(problem=model)
    if hyperparameters["c0"] == 0.0:
        hyperparameters["c0"] = return_c0(model=model)
    if hyperparameters["q"] == 0.0:
        hyperparameters["q"] = return_q(model)
        hyperparameters["r_q"] = 1.0
    else:
        hyperparameters["r_q"] = return_rx(num_iter, hyperparameters["q"], float(args.q_final))
    logfiles = {}
    for solver in solvers:
        logfiles[solver] = []
        for run in range(nb_runs):
            logfile = logpath / f"{solver}_{benchmark}_nbiter{num_iter}_run{run}.log"
            logfiles[solver].append(logfile)

    make_solvers_thread(
        solvers, sample=s_init, model=model, num_iter=num_iter, nb_runs=nb_runs, logfiles=logfiles, **hyperparameters
    )
