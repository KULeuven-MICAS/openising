import numpy as np
import os
import argparse
import openjij as oj
import pathlib

from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers

from ising.ising.utils.helper_solvers import run_solver, return_c0, return_rx
from ising.utils.numpy import triu_to_symm

TOP = pathlib.Path(os.getenv("TOP"))

parser = argparse.ArgumentParser()
parser.add_argument()
parser.add_argument("-benchmark", help="Name of the benchmark to run", default="K2000")
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=10)
parser.add_argument("--num_iter", help="Range for number of iterations", default=(1000, 5000), nargs="+")
parser.add_argument("-figName", help="Name of the figure that needs to be saved", default="Energy_accuracy_check.png")

# BRIM parameters
parser.add_argument("-dtBRIM", help="Time step for BRIM", default=1e-6)
parser.add_argument("-C", help="capacitor parameter", default=1e-5)
parser.add_argument("-kmin", help="Minimum latch strength", default=0.01)
parser.add_argument("-kmax", help="Maximum latch strength", default=3)
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
parser.add_argument("-a0", help="Bifurcation parameter", default=1.0)

args = parser.parse_args()


benchmark = args.benchmark
print("Generating benchmark: ", benchmark)
graph, best_found = G_parser(benchmark=TOP / f"ising/benchmarks/G/{benchmark}.txt")
model = MaxCut(graph=graph)
print("Generated benchmark")

solvers = list(args.solvers)
if solvers[0] == "all":
    solvers = ["BRIM", "SA", "SCA", "bSB", "dSB"]

num_iter = tuple(args.num_iter)
nb_runs = int(args.nb_runs)
iter_list = np.linspace(num_iter[0], num_iter[1], nb_runs, dtype=int)

print("Retrieving parameter info")
# BRIM params
dtBRIM = float(args.dtBRIM)
C = float(args.C)
G = np.average(np.sum(np.abs(triu_to_symm(model.J)), axis=0)) * 2
print("G: ", str(G))
kmin = float(args.kmin)
kmax = float(args.kmax)
flip = bool(args.flip)

# SA params
T = float(args.T)
Tfin = float(args.Tfin)
seed = int(args.seed)
np.random.seed(seed)

# SCA params
q = float(args.q)
qfin = float(args.q_final)

# SB params
dt = float(args.BRIM)
a0 = float(args.a0)
c0 = return_c0(model=model)

print("Setting up solvers")
logfiles = dict()
logpath = TOP / "ising/flow/logs"
figpath = TOP / "ising/flow/plots/benchmark_solvers"

for num_iter in iter_list:
    logfiles[num_iter] = dict()
    s_init = np.random.choice([-1, 1], (model.num_variables,))
    r_T = return_rx(num_iter=num_iter, r_init=T, r_final=Tfin)
    r_q = return_rx(num_iter=num_iter, r_init=q, r_final=qfin)
    at = lambda t: a0 / (dt * num_iter) * t

    for solver in solvers:
        logfiles[num_iter][solver] = []
        for run in range(nb_runs):
            print(f"Run {run} for {solver} with {num_iter} iterations")
            logfile = logpath / f"{solver}_nbiter{num_iter}_run{run}.log"
            run_solver(
                solver,
                num_iter=num_iter,
                s_init=s_init,
                logfile=logfile,
                model=model,
                dtBRIM=dtBRIM, kmin=kmin, kmax=kmax, C=C, G=G, flip=flip, seed=seed, # BRIM parameters
                T=T, r_T=r_T, q=q, r_q=r_q,                                          # SA and SCA parameters
                dtSB=dt, a0=a0, at=at, c0=c0                                         # SB parameters
            )
            logfiles[num_iter][solver].append(logfile)

plot_energy_dist_multiple_solvers(
    logfiles,
    xlabel="Number of iterations",
    figName="energy_dist_iter.png",
    best_found=np.ones((len(iter_list),)) * best_found,
    save_folder=figpath,
)
