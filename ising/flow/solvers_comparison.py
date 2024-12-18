import os
import pathlib
import numpy as np
import argparse
import openjij as oj
import matplotlib.pyplot as plt

from ising.generators.MaxCut import random_MaxCut

# import all solvers for comparison
from ising.solvers.BRIM import BRIM
from ising.solvers.DSA import DSASolver
from ising.solvers.SA import SASolver
from ising.solvers.SB import ballisticSB, discreteSB
from ising.solvers.SCA import SCA
from ising.solvers.exhaustive import ExhaustiveSolver

from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers
from ising.postprocessing.plot_solutions import plot_state_continuous, plot_state_discrete

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
C = float(args.C)
G = float(args.G)
kmin = float(args.kmin)
kmax = float(args.kmax)
flip = bool(args.flip)

# SA parameters
T = float(args.T)
Tfin = float(args.Tfin)
r_T = (Tfin / T) ** (1 / (num_iter + 1))

# SCA parameters
q = float(args.q)
r_q = (float(args.q_final) / q) ** (1 / (num_iter + 1))

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
    v = np.random.choice([-0.5, 0.5], (N,))
    x = np.random.uniform(-0.1, 0.1, (N,))
    y = np.zeros((N,))
    sigma = np.random.choice([-1.0, 1.0], (N,))

    c0 = 0.5 / (
        np.sqrt(problem.num_variables)
        * np.sqrt(np.sum(np.power(problem.J, 2)) / (problem.num_variables * (problem.num_variables - 1)))
    )

    for solver in solvers:
        logfiles[N][solver] = []
        for run in range(nb_runs):
            print(solver, " run ", run)
            if solver == "BRIM":
                logfile = logtop / f"BRIM_N{N}_run{run}.log"
                BRIM().solve(
                    model=problem,
                    v=v,
                    num_iterations=num_iter,
                    dt=tend / num_iter,
                    kmin=kmin,
                    kmax=kmax,
                    C=C,
                    G=G,
                    file=logfile,
                    random_flip=flip,
                    seed=seed
                )
                logfiles[N]["BRIM"].append(logfile)
                plot_state_continuous(logfiles[N]["BRIM"][-1], figname=f"BRIM_N{N}.png", save_folder=figtop)
            elif solver == "DSA":
                logfile = logtop / f"DSA_N{N}_run{run}.log"
                DSASolver().solve(
                    model=problem,
                    initial_state=sigma,
                    num_iterations=num_iter,
                    initial_temp=T,
                    cooling_rate=r_T,
                    file=logfile,
                    seed=int(args.seed),
                )
                logfiles[N]["DSA"].append(logfile)
                plot_state_discrete(logfiles[N]["DSA"][-1], figName=f"DSA_N{N}.png", save_folder=figtop)
            elif solver == "SA":
                logfile = logtop / f"SA_N{N}_run{run}.log"
                SASolver().solve(
                    model=problem,
                    initial_state=sigma,
                    num_iterations=num_iter,
                    initial_temp=T,
                    cooling_rate=r_T,
                    file=logfile,
                    seed=int(args.seed),
                )
                logfiles[N]["SA"].append(logfile)
                plot_state_discrete(logfiles[N]["SA"][-1], figName=f"SA_N{N}.png", save_folder=figtop)
            elif solver == "bSB":
                logfile = logtop / f"bSB_N{N}_run{run}.log"
                s_optim, energy = ballisticSB().solve(
                    model=problem,
                    x=np.copy(x),
                    y=np.copy(y),
                    num_iterations=num_iter,
                    at=at,
                    a0=1.0,
                    c0=c0,
                    dt=dt,
                    file=logfile,
                )
                logfiles[N]["bSB"].append(logfile)
                plot_state_continuous(logfiles[N]["bSB"][-1], figname=f"bSB_N{N}.png", save_folder=figtop)
                print(s_optim)
            elif solver == "dSB":
                logfile = logtop / f"dSB_N{N}_run{run}.log"
                s_optim, energy = discreteSB().solve(
                    model=problem,
                    x=np.copy(x),
                    y=np.copy(y),
                    num_iterations=num_iter,
                    at=at,
                    a0=1.0,
                    c0=c0,
                    dt=dt,
                    file=logfile,
                )
                logfiles[N]["dSB"].append(logfile)
                plot_state_continuous(logfiles[N]["dSB"][-1], figname=f"dSB_N{N}.png", save_folder=figtop)
                print(s_optim)
            elif solver == "SCA":
                logfile = logtop / f"SCA_N{N}_run{run}.log"
                SCA().solve(
                    model=problem,
                    sample=sigma,
                    num_iterations=num_iter,
                    T=T,
                    r_t=r_T,
                    q=q,
                    r_q=r_q,
                    seed=int(args.seed),
                    file=logfile,
                )
                logfiles[N]["SCA"].append(logfile)
                plot_state_discrete(logfiles[N]["SCA"][-1], figName=f"SCA_N{N}.png", save_folder=figtop)

plot_energy_dist_multiple_solvers(
    logfiles,
    xlabel="problem size",
    best_found=best_found,
    save_folder=TOP / "ising/flow/plots/Solvers_comparison",
    figName=str(args.figName),
)
