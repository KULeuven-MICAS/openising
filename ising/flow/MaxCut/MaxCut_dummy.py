import os
import pathlib
import numpy as np
import argparse
import networkx as nx
import time

from ising.generators.MaxCut import random_MaxCut

# import all solvers for comparison
from ising.solvers.Gurobi import Gurobi

from ising.utils.helper_solvers import run_solver, return_c0, return_rx, return_G, return_q
from ising.postprocessing.MC_plot import plot_MC_solution
from ising.utils.flow import make_directory, parse_hyperparameters

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument("--N_list", help="tuple containing min and max problem size",
                    default=(2, 10), nargs="+", type=tuple)
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
parser.add_argument("-nb_runs", help="Number of runs", default=3)
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
parser.add_argument("-fig_folder", help="Folder inwhich to save the figures", default="")
parser.add_argument("-fig_name", help="Name of the figure that needs to be saved", default="Energy_accuracy_check.png")

# BRIM parameters
parser.add_argument("-t_end", help="End time for the simulation", default=3e-5)
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
parser.add_argument("-q", help="initial penalty value", default=0.)
parser.add_argument("-q_final", help="final penalty value", default=10.0)

# SB parameters
parser.add_argument("-dt", help="Time step for simulated bifurcation", default=0.25)
parser.add_argument('-a0', help="Parameter a0 of SB", default=1.0)
parser.add_argument('-c0', help="Parameter c0 of SB", default=0.0)


print("parsing args")
args = parser.parse_args()
if args.solvers == "all":
    solvers = ["BRIM", "SA", "bSB", "dSB", "SCA"]
else:
    solvers = args.solvers[0].split()

Nlist = args.N_list[0].split()
nb_runs = int(args.nb_runs)
Nlist = np.linspace(Nlist[0], Nlist[1], nb_runs, dtype=int)

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
if hyperparameters["q"] == 0.:
    change_q = True
    r_q = 1.0
else:
    r_q = return_rx(num_iter, hyperparameters["q"], float(args.q_final))
    change_q = False

# SB parameters
if hyperparameters["c0"] == 0.:
    change_c = True
else:
    change_c = False



logtop = TOP / "ising/flow/MaxCut/logs"
make_directory(logtop)
figtop = TOP / "ising/flow/MaxCut/plots" / str(args.fig_folder)
make_directory(figtop)
best_found = []



problems = {}
for N in Nlist:
    problem = random_MaxCut(N)
    problems[N] = problem

for N in Nlist:
    problem = problems[N]
    print(f"Generated problem: {problem}")

    G_orig = nx.Graph()
    G_orig.add_nodes_from(list(range(N)))
    for i in range(N):
        for j in range(i+1, N):
            if problem.J[i, j] != 0:
                G_orig.add_edge(i, j)

    if change_G:
        G = return_G(problem=problem)
    if change_c:
        c0 = return_c0(model=problem)
    if change_q:
        q = return_q(problem)
        print(f"{q=}")

    if use_gurobi:
        print("Solving with Gurobi")
        logfile = logtop / f"Gurobi_N{N}.log"
        sigma_base, energy_base = Gurobi().solve(model=problem, file=logfile)
        best_found.append(energy_base)

        print(f"Gurobi best state {sigma_base}")

    sigma = np.random.choice([-1.0, 1.0], (N,), p=[0.5, 0.5])

    for solver in solvers:
        print(f"Running with {solver}")
        for run in range(nb_runs):
            logfile = logtop / f"{solver}_N{N}_run{run}.log"
            optim_state, optim_energy = run_solver(
                solver,
                num_iter=num_iter,
                s_init=sigma,
                logfile=logfile,
                model=problem,
                **hyperparameters
            )
        print(f"{solver} {optim_energy=}")
        print(f"{solver} {optim_state=}")
        plot_MC_solution(fileName=logfile, G_orig=G_orig, save_folder=figtop,
                         fig_name=f"{solver}_N{N}_graph_{fig_name}")
