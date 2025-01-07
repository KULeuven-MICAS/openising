import os
import pathlib
import numpy as np
import argparse
import openjij as oj
import matplotlib.pyplot as plt

from ising.generators.MaxCut import random_MaxCut

# import all solvers for comparison
from ising.solvers.exhaustive import ExhaustiveSolver

from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers
from ising.utils.helper_solvers import run_solver, return_c0, return_rx


from ising.utils.numpy import triu_to_symm

TOP = pathlib.Path(os.getenv("TOP"))
parser = argparse.ArgumentParser()
parser.add_argument("--Nlist", help="tuple containing min and max problem size", default=(10, 100), nargs="+", type=int)
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("-num_iter", help="Number of iterations for each run", default=1000)
parser.add_argument("-figName", help="Name of the figure that needs to be saved", default="Energy_accuracy_check.png")
parser.add_argument("-plot", help="Whether to plot the results", default=False)

# BRIM parameters
parser.add_argument("-tend", help="End time for the simulation", default=3e-5)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-G", help="Resistor parameter", default=1e-1)
parser.add_argument("-kmin", help="Minimum latch strength", default=0.01)
parser.add_argument("-kmax", help="Maximum latch strength", default=2.5)
parser.add_argument("-flip", help="Whether to activate random flipping in BRIM", default=False)

# SA parameters
parser.add_argument("-T", help="Initial temperature", default=50.0)
parser.add_argument("-Tfin", help="Final temperature of the annealing process", default=0.05)
parser.add_argument("-seed", help="Seed for random number generator", default=1)

# SCA parameters
parser.add_argument("-q", help="initial penalty value", default=5.0)
parser.add_argument("-q_final", help="final penalty value", default=10.0)

# SB parameters
parser.add_argument("-dt", help="Time step for simulated bifurcation", default=0.25)


print("parsing args")
args = parser.parse_args()
if args.solvers == "all":
    solvers = ["BRIM", "DSA", "SA", "bSB", "dSB", "SCA"]
else:
    solvers = list(args.solvers)

Nlist = tuple(args.Nlist)
nb_runs = int(args.nb_runs)
Nlist = np.linspace(Nlist[0], Nlist[1], nb_runs, dtype=int)
seed = int(args.seed)
np.random.seed(seed)
if bool(args.plot):
    plt.ion()

num_iter = int(args.num_iter)

# BRIM parameters
tend = float(args.tend)
dtBRIM = tend / num_iter
C = float(args.C)
G = float(args.G)
kmin = float(args.kmin)
kmax = float(args.kmax)
flip = bool(args.flip)

# SA parameters
T = float(args.T)
Tfin = float(args.Tfin)
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
for N in Nlist:
    logfiles[N] = dict()
    print("Generating MaxCut problem of size", N)
    problem = random_MaxCut(N)
    sumJ = np.sum(np.abs(triu_to_symm(problem.J)), axis=0)
    G = np.average(sumJ)*2
    print("generated problem", problem)
    print("G: ", G)

    if N < 30:
        print("Solving with Exhaustive solver")
        best_state, energy_best = ExhaustiveSolver().solve(model=problem, file=None)
        best_found.append(energy_best)
        print("Found best solution: ", best_state)
    else:
        print("Solving with OpenJij")
        bqm = oj.BinaryQuadraticModel.from_numpy_matrix(np.diag(problem.h) - triu_to_symm(problem.J))
        sampler = oj.SASampler()
        response = sampler.sample(bqm, num_reads=nb_runs)
        best_found.append(response.first.energy)
        print("Found best solution: ", response.first.sample)
    sigma = np.random.choice([-1.0, 1.0], (N,))

    c0 = return_c0(model=problem)

    for solver in solvers:
        logfiles[N][solver] = []
        for run in range(nb_runs):
            print(solver, " run ", run)
            logfile = logtop / f"{solver}_N{N}_run{run}.log"
            run_solver(
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


plot_energy_dist_multiple_solvers(
    logfiles,
    xlabel="problem size",
    #best_found=best_found,
    save_folder=TOP / "ising/flow/plots/Solvers_comparison",
    figName=str(args.figName),
)
