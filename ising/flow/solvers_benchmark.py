import numpy as np
import os
import argparse
import pathlib

from ising.benchmarks.parsers.G import G_parser
from ising.generators.MaxCut import MaxCut
from ising.postprocessing.energy_plot import plot_energy_dist_multiple_solvers, plot_energy_time_multiple
from ising.postprocessing.plot_solutions import plot_state_continuous

from ising.utils.helper_solvers import run_solver, return_c0, return_rx
from ising.utils.numpy import triu_to_symm

TOP = pathlib.Path(os.getenv("TOP"))

parser = argparse.ArgumentParser()
parser.add_argument("-benchmark", help="Name of the benchmark to run", default="G1")
parser.add_argument("--solvers", help="Which solvers to run", default="all", nargs="+")
parser.add_argument("-nb_runs", help="Number of runs", default=5)
parser.add_argument("--num_iter", help="Range for number of iterations", default=(1000, 5000), nargs="+")
parser.add_argument("-clock_freq", help="Frequency of the clock", default=1e6)
parser.add_argument("-clock_op", help="Amount of operations per clock cycle", default=1000)

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
if best_found is not None:
    print("Best found energy: ", -best_found)
print("Generated benchmark")

if args.solvers == "all":
    solvers = ["SA", "SCA", "bSB", "dSB", "BRIM"]
else:
    solvers = args.solvers
print("Solving with following solvers: ", solvers)

num_iter = tuple(args.num_iter)
nb_runs = int(args.nb_runs)
iter_list = np.linspace(int(num_iter[0]), int(num_iter[1]), nb_runs, dtype=int)

clock_freq = float(args.clock_freq)
clock_op = int(args.clock_op)

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
dt = float(args.dt)
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
                clock_freq=clock_freq,
                clock_op=clock_op,
                dtBRIM=dtBRIM, kmin=kmin, kmax=kmax, C=C, G=G, flip=flip, seed=seed,  # BRIM parameters
                T=T, r_T=r_T, q=q, r_q=r_q,  # SA and SCA parameters
                dtSB=dt, a0=a0, at=at, c0=c0,  # SB parameters
            )
            logfiles[num_iter][solver].append(logfile)
        if solver == "BRIM" or solver == "bSB" or solver == "dSB":
            plot_state_continuous(
                logfiles[num_iter][solver][-1], figname=f"state_iter{num_iter}_solver{solver}_benchmark{benchmark}.png",
                save_folder=figpath
            )
if best_found is not None:
    best_found = -best_found

print("Plotting energy distribution of solver in function of number of iterations and time")
plot_energy_dist_multiple_solvers(
    logfiles,
    xlabel="Number of iterations",
    figName=f"energy_dist_iter_{benchmark}.png",
    best_found=[best_found]*len(iter_list),
    save_folder=figpath,
)
plot_energy_time_multiple(logfiles, best_found, save_folder=figpath, figName=f"energy_time_{benchmark}.png")
