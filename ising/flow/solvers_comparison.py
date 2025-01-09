import os
import pathlib
import numpy as np
import argparse
import networkx as nx
import time
# import threading


from ising.generators.MaxCut import random_MaxCut

# import all solvers for comparison
from ising.solvers.Gurobi import Gurobi

from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers
from ising.postprocessing.plot_solutions import plot_state_continuous, plot_state_discrete
from ising.utils.helper_solvers import run_solver, return_c0, return_rx, return_G
from ising.postprocessing.MC_plot import plot_MC_solution

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument("--N_list", help="tuple containing min and max problem size",
                    default=(2, 10), nargs="+", type=tuple)
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-use_gurobi", help="Whether to use Gurobi as baseline", default=False)
parser.add_argument("-nb_runs", help="Number of runs", default=3)
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
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
parser.add_argument("-seed", help="Seed for random number generator", default=1)

# SCA parameters
parser.add_argument("-q", help="initial penalty value", default=5.0)
parser.add_argument("-q_final", help="final penalty value", default=10.0)

# SB parameters
parser.add_argument("-dt", help="Time step for simulated bifurcation", default=0.25)


print("parsing args")
args = parser.parse_args()
if args.solvers == "all":
    solvers = ["BRIM", "SA", "bSB", "dSB", "SCA"]
else:
    solvers = list(args.solvers)

Nlist = tuple(args.N_list)
nb_runs = int(args.nb_runs)
Nlist = np.linspace(Nlist[0], Nlist[1], nb_runs, dtype=int)
seed = int(args.seed)
if seed == 0:
    seed = int(time.time())
np.random.seed(seed)
print(seed)
fig_name = str(args.fig_name)
use_gurobi = bool(args.use_gurobi)

num_iter = int(args.num_iter)

# BRIM parameters
tend = float(args.t_end)
dtBRIM = tend / num_iter
C = float(args.C)
G = float(args.G)

kmin = float(args.k_min)
kmax = float(args.k_max)
flip = bool(args.flip)

# SA parameters
T = float(args.T)
Tfin = float(args.T_final)
r_T = return_rx(num_iter, T, Tfin)

# SCA parameters
q = float(args.q)
r_q = return_rx(num_iter, q, float(args.q_final))

# SB parameters
dt = float(args.dt)

def at(t):
    return 1.0 / (dt * num_iter) * t


logfiles = dict()
logtop = TOP / "ising/flow/logs"
figtop = TOP / "ising/flow/plots/Solvers_comparison"
best_found = []

if G == 0:
    change_G = True
else:
    change_G = False

problems = {}
for N in Nlist:
    problem = random_MaxCut(N)
    problems[N] = problem

for N in Nlist:
    logfiles[N] = dict()
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
    if use_gurobi:
        print("Solving with Gurobi")
        sigma_base, energy_base = Gurobi().solve(model=problem)
        _, c = problem.to_qubo()
        best_found.append(energy_base+c)

        print(f"Gurobi best state {sigma_base}")

    sigma = np.random.choice([-1.0, 1.0], (N,), p=[0.5, 0.5])
    c0 = return_c0(model=problem)

    for solver in solvers:
        logfiles[N][solver] = []
        print(f"Running with {solver}")
        for run in range(nb_runs):
            logfile = logtop / f"{solver}_N{N}_run{run}.log"
            optim_state, optim_energy = run_solver(
                solver,
                num_iter=num_iter,
                s_init=sigma,
                logfile=logfile,
                model=problem,
                dtBRIM=dtBRIM, kmin=kmin, kmax=kmax, C=C, G=G, flip=flip, seed=seed,  # BRIM parameters
                T=T, r_T=r_T, q=q, r_q=r_q,  # SA and SCA parameters
                dtSB=dt, a0=1., at=at, c0=c0,  # SB parameters
            )
            logfiles[N][solver].append(logfile)
        print(f"{solver} {optim_energy=}")
        print(f"{solver} {optim_state=}")
        plot_MC_solution(fileName=logfiles[N][solver][-1], G_orig=G_orig, save_folder=figtop,
                         fig_name=f"{solver}_N{N}_graph_{fig_name}")
        if solver in ["BRIM", "bSB", "dSB"]:
            plot_state_continuous(logfile=logfiles[N][solver][-1], figname=f"{solver}_N{N}_{fig_name}",
                                   save_folder=figtop)
        else:
            plot_state_discrete(logfile=logfiles[N][solver][-1], figname=f"{solver}_N{N}_{fig_name}",
                                 save_folder=figtop)


plot_energy_dist_multiple_solvers(
    logfiles,
    xlabel="problem size",
    best_found=best_found if use_gurobi else None,
    best_Gurobi=use_gurobi,
    save_folder=TOP / "ising/flow/plots/Solvers_comparison",
    figName=f"{solver}_{fig_name}",
)
