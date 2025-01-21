import os
import pathlib
import numpy as np
import argparse
import networkx as nx
import time

from ising.generators.MaxCut import random_MaxCut

from ising.utils.helper_solvers import return_c0, return_rx, return_G, return_q
from ising.postprocessing.MC_plot import plot_MC_solution
from ising.utils.flow import make_directory, parse_hyperparameters
from ising.utils.threading import make_solvers_thread, make_Gurobi_thread

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument(
    "--N_list", help="tuple containing min and max problem size", default=(2, 10), nargs="+"
)
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
parser.add_argument("-nb_runs", help="Number of runs", default=3)
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
parser.add_argument("-fig_folder", help="Folder inwhich to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default="Energy_accuracy_check.png")

# BRIM parameters
parser.add_argument("-dtBRIM", help="End time for the simulation", default=3e-9)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("-k_min", help="Minimum latch strength", default=0.01)
parser.add_argument("-k_max", help="Maximum latch strength", default=2.5)
parser.add_argument("-flip", help="Whether to activate random flipping in BRIM", default=False)

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


print("parsing args")
args = parser.parse_args()
if args.solvers == "all":
    solvers = ["BRIM", "SA", "bSB", "dSB", "SCA"]
else:
    solvers = args.solvers[0].split()

Nlist = args.N_list[0].split()
nb_runs = int(args.nb_runs)
Nlist = np.array(range(int(Nlist[0]), int(Nlist[1]), 10))

seed = int(args.seed)
if seed == 0:
    seed = int(time.time())
np.random.seed(seed)

fig_name = str(args.fig_name)
use_gurobi = bool(args.use_gurobi)

num_iter = int(args.num_iter)
hyperparameters = parse_hyperparameters(args, num_iter)

# BRIM parameters
if hyperparameters["G"] == 0:
    change_G = True
else:
    change_G = False

# SCA parameters
if hyperparameters["q"] == 0.0:
    change_q = True
    hyperparameters["r_q"] = 1.0
else:
    hyperparameters["r_q"] = return_rx(num_iter, hyperparameters["q"], float(args.q_final))
    change_q = False

# SB parameters
if hyperparameters["c0"] == 0.0:
    change_c = True
else:
    change_c = False


logtop = TOP / "ising/flow/MaxCut/logs"
make_directory(logtop)
figtop = TOP / "ising/flow/MaxCut/plots" / str(args.fig_folder)
make_directory(figtop)


problems = {}
for N in Nlist:
    problem = random_MaxCut(N)
    problems[N] = problem

if use_gurobi:
    logfiles = {}
    for N in Nlist:
        logfile = logtop / f"Gurobi_N{N}.log"
        logfiles[N] = logfile
    make_Gurobi_thread(nb_cores=3, models=problems, logfiles=logfiles)


for N in Nlist:
    problem = problems[N]
    if change_G:
        hyperparameters["G"] = return_G(problem=problem)
    if change_c:
        hyperparameters["c0"] = return_c0(model=problem)
    if change_q:
        hyperparameters["q"] = return_q(problem)
    logfiles = {}
    print(np.sum(problem.J, axis=1))
    for solver in solvers:
        logfiles[solver] = []
        for nb_run in range(nb_runs):
            logfiles[solver].append(logtop / f"{solver}_N{N}_nb_run{nb_run}.log")
    sigma = np.random.choice([-1.0, 1.0], (N,), p=[0.5, 0.5])
    make_solvers_thread(
        nb_cores=len(solvers),
        solvers=solvers,
        sample=sigma,
        num_iter=num_iter,
        model=problem,
        nb_runs=nb_runs,
        logfiles=logfiles,
        **hyperparameters,
    )
    if N <= 20:
        G_orig = nx.Graph()
        G_orig.add_nodes_from(list(range(N)))
        for i in range(N):
            for j in range(i+1, N):
                if problem.J[i, j] != 0:
                    G_orig.add_edge(i, j)

        for solver in solvers:
            plot_MC_solution(fileName=logfiles[solver][-1], G_orig=G_orig, save_folder=figtop,
                            fig_name=f"{solver}_N{N}_graph_{fig_name}")
